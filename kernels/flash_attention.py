"""
FlashAttention-2 Forward Pass — Triton kernel

The key insight: instead of materializing the full (seq_len, seq_len) attention
matrix in HBM, tile over K/V blocks and compute attention incrementally using
the online softmax trick.

Memory: O(N) instead of O(N^2)
Speed: fewer HBM accesses due to tiling

Online softmax trick:
  Normal softmax needs all scores to compute max and sum.
  Online softmax maintains a running max (m) and running sum (l),
  correcting previous results when a new tile has a larger max.

  For each new tile of scores S:
    m_new = max(m_old, max(S))
    correction = exp(m_old - m_new)
    l_new = correction * l_old + sum(exp(S - m_new))
    acc_new = correction * acc_old + exp(S - m_new) @ V_tile

  At the end: output = acc / l
"""

import torch
import triton
import triton.language as tl

TILE_SIZE = 64


def flash_attention_pytorch(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Reference: same result as naive attention, but implemented with
    the tiled online softmax algorithm in PyTorch for debugging.

    Q, K, V: (seq_len, d_k)
    """
    seq_len, d_k = Q.shape
    output = torch.zeros_like(Q, device=Q.device, dtype=Q.dtype)

    for i in range(seq_len):
        m = torch.tensor([float("-inf")], device=Q.device, dtype=Q.dtype)
        l = 0
        acc = torch.zeros((d_k,), device=Q.device, dtype=Q.dtype)
        q = Q[i, :]
        for j in range(0, seq_len, TILE_SIZE):
            k_tile = K[j : j + TILE_SIZE, :]
            v_tile = V[j : j + TILE_SIZE, :]
            scores = q @ k_tile.T * (d_k**-0.5)
            m_new = torch.max(scores.max(), m)
            correction = torch.exp(m - m_new)
            scores_exp = (scores - m_new).exp()
            l_new = correction * l + scores_exp.sum()
            acc = correction * acc + scores_exp @ v_tile

            # update
            m = m_new
            l = l_new
        output[i, :] = acc / l

    return output


def flash_attention_naive(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """Naive attention for correctness comparison."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / (d_k**0.5)
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return weights @ V


@triton.jit
def flash_attention_kernel(
    Q,
    K,
    V,
    Output,
    seq_len,
    d_k,
    stride_q,
    stride_k,
    stride_v,
    stride_o,
    BLOCK_SEQ: tl.constexpr,  # tile size for K/V sequence dimension
    BLOCK_DK: tl.constexpr,  # block size for d_k (head dimension)
):
    """Single-row FlashAttention kernel (kept for reference)."""
    pid = tl.program_id(axis=0)

    block_start_q = pid * stride_q
    offsets_col = tl.arange(0, BLOCK_DK)
    mask_col = offsets_col < d_k

    offsets_q = block_start_q + offsets_col

    q = tl.load(Q + offsets_q, mask_col)

    m = float("-inf")
    l = 0.0
    acc = tl.zeros((BLOCK_DK,), dtype=tl.float32)

    offsets_row = tl.arange(0, BLOCK_SEQ)

    for start in tl.range(0, seq_len, BLOCK_SEQ):
        offsets_row_i = offsets_row + start
        offsets_2d_i = offsets_row_i[:, None] * stride_k + offsets_col[None, :]

        mask_row_i = offsets_row_i < seq_len
        mask_2d_i = mask_row_i[:, None] * mask_col[None, :]

        k_tile = tl.load(K + offsets_2d_i, mask_2d_i)
        v_tile = tl.load(V + offsets_2d_i, mask_2d_i)

        scores = tl.sum(q[None, :] * k_tile, axis=1) / tl.sqrt(d_k.to(tl.float32))

        m_new = tl.maximum(m, tl.max(scores))
        correction = tl.exp(m - m_new)
        scores_exp = tl.exp(scores - m_new)
        l_new = l * correction + tl.sum(scores_exp)
        acc = acc * correction + tl.sum(scores_exp[:, None] * v_tile, axis=0)

        m = m_new
        l = l_new
    acc = acc / l

    offsets_o = pid * stride_o + offsets_col
    tl.store(Output + offsets_o, acc.to(tl.float16), mask_col)


@triton.jit
def flash_attention_v2_kernel(
    Q,
    K,
    V,
    Output,
    seq_len,
    d_k,
    stride_q,
    stride_k,
    stride_v,
    stride_o,
    BLOCK_Q: tl.constexpr,  # tile size for Q rows per program
    BLOCK_SEQ: tl.constexpr,  # tile size for K/V sequence dimension
    BLOCK_DK: tl.constexpr,  # block size for d_k (head dimension)
):
    """
    Full FlashAttention-2: each program handles BLOCK_Q query rows.
    Uses tl.dot for tensor core acceleration on score and output matmuls.
    """
    pid = tl.program_id(axis=0)

    block_q = pid * stride_q * BLOCK_Q

    offsets_row = tl.arange(0, BLOCK_Q)
    mask_row = offsets_row < seq_len
    offsets_col = tl.arange(0, BLOCK_DK)
    mask_col = offsets_col < d_k

    offsets = block_q + offsets_row[:, None] * stride_q + offsets_col[None, :]
    mask = mask_row[:, None] * mask_col[None, :]

    q = tl.load(Q + offsets, mask)

    m = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_DK), dtype=tl.float32)

    offsets_row_kv = tl.arange(0, BLOCK_SEQ)

    for start in tl.range(0, seq_len, BLOCK_SEQ):
        offsets_row_i = offsets_row_kv + start
        offsets_2d_i = offsets_row_i[:, None] * stride_k + offsets_col[None, :]

        mask_row_i = offsets_row_i < seq_len
        mask_2d_i = mask_row_i[:, None] * mask_col[None, :]

        k_tile = tl.load(K + offsets_2d_i, mask_2d_i)
        v_tile = tl.load(V + offsets_2d_i, mask_2d_i)

        scores = tl.dot(q, tl.trans(k_tile)) / tl.sqrt(d_k.to(tl.float32))

        m_new = tl.maximum(m, tl.max(scores, axis=-1))
        correction = tl.exp(m - m_new)
        scores_exp = tl.exp(scores - m_new[:, None])
        l_new = l * correction + tl.sum(scores_exp, axis=-1)
        acc = acc * correction[:, None] + tl.dot(scores_exp, v_tile.to(tl.float32))

        m = m_new
        l = l_new
    acc = acc / l[:, None]

    tl.store(Output + offsets, acc.to(tl.float16), mask)


def flash_attention_triton(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    seq_len, d_k = Q.shape
    stride = d_k

    output = torch.empty_like(Q)

    # A100-optimized block sizes
    BLOCK_Q = 64
    BLOCK_SEQ = 64
    BLOCK_DK = 64

    grid = (triton.cdiv(seq_len, BLOCK_Q),)
    flash_attention_v2_kernel[grid](
        Q,
        K,
        V,
        output,
        seq_len,
        d_k,
        stride_q=stride,
        stride_k=stride,
        stride_v=stride,
        stride_o=stride,
        BLOCK_Q=BLOCK_Q,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DK=BLOCK_DK,
    )

    return output
