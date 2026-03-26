"""
Naive Attention — PyTorch reference + Triton kernel

attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Single-head, no masking. Each program handles one row of the output
(one query token's attention over all key/value tokens).

This only works for short sequences where the full attention row fits in BLOCK_SIZE.
FlashAttention removes this limitation via tiling.
"""

import torch
import triton
import triton.language as tl


def attention_pytorch(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:

    d_k = Q.shape[-1]
    w = Q @ K.T / (d_k**0.5)
    return torch.nn.functional.softmax(w, dim=-1) @ V


def attention_native(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
    ).squeeze(0)


@triton.jit
def attention_kernel(
    Q,  # (seq_len, d_k)
    K,  # (seq_len, d_k)
    V,  # (seq_len, d_k)
    Output,  # (seq_len, d_k)
    seq_len,  # number of tokens
    d_k,  # head dimension
    stride_q,  # row stride of Q
    stride_k,  # row stride of K
    stride_v,  # row stride of V
    stride_o,  # row stride of Output
    BLOCK_DK: tl.constexpr,  # block size for d_k dimension
    BLOCK_SEQ: tl.constexpr,  # block size for seq_len dimension
):
    """
    Naive attention kernel
    """
    pid = tl.program_id(axis=0)

    row_offsets = tl.arange(0, BLOCK_SEQ)
    col_offsets = tl.arange(0, BLOCK_DK)

    # redundant, strides are the same for this simplifed implementation
    offsets_q = pid * stride_q + col_offsets
    offsets_o = pid * stride_o + col_offsets

    offsets_2d = row_offsets[:, None] * stride_k + col_offsets[None, :]

    mask_row = row_offsets < seq_len
    mask_col = col_offsets < d_k
    mask_2d = mask_row[:, None] * mask_col[None, :]

    q = tl.load(Q + offsets_q, mask_col)
    k_block = tl.load(K + offsets_2d, mask_2d)
    v_block = tl.load(V + offsets_2d, mask_2d)

    # Q row @ K.T
    scores = tl.sum(q[None, :] * k_block, axis=1) / tl.sqrt(d_k.to(tl.float32))
    scores = tl.where(mask_row, scores, float("-inf"))
    scores = tl.softmax(scores)
    output = tl.sum(scores[:, None] * v_block, axis=0)

    tl.store(Output + offsets_o, output.to(tl.float16), mask_col)


def attention_triton(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    seq_len, stride = Q.shape

    output = torch.empty_like(Q)

    grid = (seq_len,)
    attention_kernel[grid](
        Q,
        K,
        V,
        output,
        seq_len,
        stride,
        stride_q=stride,
        stride_k=stride,
        stride_v=stride,
        stride_o=stride,
        BLOCK_SEQ=64,  # hardcoded for basic test case
        BLOCK_DK=64,
    )

    return output
