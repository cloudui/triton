"""
FlashAttention-2 — Batched, Multi-Head, Causal

Full-featured FlashAttention with:
  - Batch dimension
  - Multi-head attention
  - Optional causal masking (autoregressive)

Q, K, V: (batch, n_heads, seq_len, d_k)
Output:   (batch, n_heads, seq_len, d_k)
"""

import torch
import triton
import triton.language as tl


def flash_attention_full_naive(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """
    Naive batched multi-head attention for correctness comparison.

    Q, K, V: (batch, n_heads, seq_len, d_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k**0.5)
    if causal:
        seq_len = Q.shape[-2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return weights @ V


def flash_attention_full_native(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """PyTorch's built-in sdpa for comparison."""
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)


@triton.jit
def flash_attention_full_kernel(
    Q,
    K,
    V,
    Output,
    seq_len,
    d_k,
    n_heads,
    stride_b,
    stride_h,
    stride_s,
    stride_d,  # shared strides (all tensors same layout)
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    """
    TODO: Implement batched multi-head FlashAttention-2 with optional causal masking.

    Grid: (num_q_blocks, batch * n_heads)
      - pid_q = tl.program_id(0)  → which Q tile
      - pid_bh = tl.program_id(1) → which batch/head (flattened)

    Steps:
    1. Compute batch/head offsets from pid_bh
       - Offset Q, K, V, Output pointers to the right batch/head slice
    2. Load Q tile — same as v2 kernel
    3. Loop over K/V tiles — same online softmax
    4. Causal masking: if IS_CAUSAL, after computing scores, mask positions
       where key_pos > query_pos with -inf before softmax
       - query positions: pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
       - key positions: start + tl.arange(0, BLOCK_SEQ)
       - mask: query_pos[:, None] >= key_pos[None, :]
       - Also: can skip entire K/V tiles where all keys are in the future
    5. Store output
    """
    pid_q = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    batch_idx = pid_bh // n_heads
    head_idx = pid_bh % n_heads

    # offset all pointers to the right batch/head slice
    bh_offset = batch_idx * stride_b + head_idx * stride_h
    Q += bh_offset
    K += bh_offset
    V += bh_offset
    Output += bh_offset

    # from here, identical to v2 kernel — just working on (seq_len, d_k) slice
    offsets_row = tl.arange(0, BLOCK_Q)
    offsets_col = tl.arange(0, BLOCK_DK)
    mask_row = offsets_row < seq_len
    mask_col = offsets_col < d_k

    q_start = pid_q * BLOCK_Q
    offsets_q = (q_start + offsets_row)[:, None] * stride_s + offsets_col[
        None, :
    ] * stride_d
    mask_q = mask_row[:, None] * mask_col[None, :]

    q = tl.load(Q + offsets_q, mask_q)

    m = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_DK), dtype=tl.float32)

    offsets_row_kv = tl.arange(0, BLOCK_SEQ)

    if IS_CAUSAL:
        end = tl.minimum(BLOCK_Q * (pid_q + 1), seq_len)
    else:
        end = seq_len

    for start in tl.range(0, end, BLOCK_SEQ):
        offsets_row_i = offsets_row_kv + start
        offsets_2d_i = (
            offsets_row_i[:, None] * stride_s + offsets_col[None, :] * stride_d
        )

        mask_row_i = offsets_row_i < seq_len
        mask_2d_i = mask_row_i[:, None] * mask_col[None, :]

        k_tile = tl.load(K + offsets_2d_i, mask_2d_i)
        v_tile = tl.load(V + offsets_2d_i, mask_2d_i)

        scores = tl.dot(q, tl.trans(k_tile)) / tl.sqrt(d_k.to(tl.float32))

        if IS_CAUSAL:
            q_positions = q_start + offsets_row
            k_positions = start + offsets_row_kv
            causal_mask = q_positions[:, None] >= k_positions[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=-1))
        correction = tl.exp(m - m_new)
        scores_exp = tl.exp(scores - m_new[:, None])
        l_new = l * correction + tl.sum(scores_exp, axis=-1)
        acc = acc * correction[:, None] + tl.dot(scores_exp, v_tile.to(tl.float32))

        m = m_new
        l = l_new
    acc = acc / l[:, None]

    tl.store(Output + offsets_q, acc.to(tl.float16), mask_q)


def flash_attention_full_triton(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """
    TODO: Wrapper to launch the kernel.

    Q, K, V: (batch, n_heads, seq_len, d_k)

    Hints:
    - Use Q.stride() to get (stride_qb, stride_qh, stride_qs, stride_qd)
    - Grid is 2D: (num_q_blocks, batch * n_heads)
    - Pass IS_CAUSAL as a constexpr so the compiler can optimize it away
    """
    n_batch, n_heads, seq_len, d_k = Q.shape
    stride_b, stride_h, stride_s, stride_d = Q.stride()

    output = torch.empty_like(Q)

    BLOCK_Q = 64
    BLOCK_SEQ = 64
    BLOCK_DK = 64

    grid = (triton.cdiv(seq_len, BLOCK_Q), n_batch * n_heads)
    flash_attention_full_kernel[grid](
        Q,
        K,
        V,
        output,
        seq_len,
        d_k,
        n_heads,
        stride_b,
        stride_h,
        stride_s,
        stride_d,
        IS_CAUSAL=causal,
        BLOCK_Q=BLOCK_Q,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DK=BLOCK_DK,
    )

    return output
