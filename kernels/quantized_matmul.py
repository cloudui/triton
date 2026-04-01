"""
Quantized Matrix Multiplication — int8 and int4 with Triton

The idea: store weights in low precision (int8/int4) to save memory,
dequantize on the fly during matmul to fp16 for computation.

fp16 matmul:  x (M, K) @ W (K, N) → out (M, N)
              W is fp16 → 2 bytes per element

int8 matmul:  x (M, K) @ W_int8 (K, N) → out (M, N)
              W stored as int8 → 1 byte per element (2x memory savings)
              Dequantize: W_fp16 = (W_int8 - zero_point) * scale

int4 matmul:  x (M, K) @ W_int4_packed (K, N//2) → out (M, N)
              Two int4 values packed per byte → 0.5 bytes per element (4x savings)
              Group quantization: each group of GROUP_SIZE weights shares scale/zero_point
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Quantization utilities (run on CPU/GPU before kernel launch)
# ============================================================


def quantize_int8(W: torch.Tensor):
    """
    Quantize fp16 weights to uint8 with per-column scale and zero_point.

    For each column j:
      scale[j] = (max(W[:, j]) - min(W[:, j])) / 255
      zero_point[j] = round(-min(W[:, j]) / scale[j])
      W_int8[i, j] = round(W[i, j] / scale[j]) + zero_point[j]

    To recover: W[i, j] ≈ (W_int8[i, j] - zero_point[j]) * scale[j]

    Returns:
      W_int8:     (K, N) uint8 — quantized weights
      scale:      (N,) float16 — one scale per column
      zero_point: (N,) float16 — one zero point per column (stored as float for convenience)
    """
    w_min = W.min(dim=0).values  # (N,)
    w_max = W.max(dim=0).values  # (N,)
    scale = (w_max - w_min) / 255.0  # (N,)
    zero_point = (
        (-w_min / scale).round().clamp(0, 255)
    )  # (N,) — which integer maps to 0.0
    W_int8 = (W / scale + zero_point).round().clamp(0, 255).to(torch.uint8)
    return W_int8, scale.to(torch.float16), zero_point.to(torch.float16)


def dequantize_int8(
    W_int8: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
):
    """Dequantize: W_fp ≈ (W_int8 - zero_point) * scale"""
    return (W_int8.float() - zero_point.float()) * scale.float()


def quantize_int4(W: torch.Tensor, group_size: int = 128):
    """
    Quantize fp16 weights to int4 with group-wise scale and zero_point.
    Two int4 values packed into one uint8.

    Same formula as int8 but:
      - 4-bit range: 0-15 (16 values) instead of 0-255
      - Scale/zero_point are per group of `group_size` rows, not per column
      - Two int4 values packed into one byte

    W: (K, N)
    Returns:
      W_packed: (K, N//2) uint8 — two int4 values packed along N dimension
      scales:   (K // group_size, N) float16 — one scale per group per column
      zeros:    (K // group_size, N) float16 — one zero point per group per column
    """
    K, N = W.shape
    assert K % group_size == 0
    assert N % 2 == 0

    W_groups = W.reshape(K // group_size, group_size, N)
    w_min = W_groups.min(dim=1).values  # (num_groups, N)
    w_max = W_groups.max(dim=1).values

    scales = (w_max - w_min) / 15.0
    zeros = (-w_min / scales).round().clamp(0, 15)

    # quantize: int_val = round(fp_val / scale) + zero_point
    W_int4 = (
        (W_groups / scales.unsqueeze(1) + zeros.unsqueeze(1))
        .round()
        .clamp(0, 15)
        .to(torch.uint8)
    )
    W_int4 = W_int4.reshape(K, N)

    # pack two int4 values along N: even columns in low nibble, odd columns in high
    W_packed = (W_int4[:, 0::2] & 0xF) | ((W_int4[:, 1::2] & 0xF) << 4)

    return W_packed, scales.to(torch.float16), zeros.to(torch.float16)


def dequantize_int4(
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
):
    """Dequantize: W_fp ≈ (int4_val - zero_point) * scale"""
    K = W_packed.shape[0]
    N_half = W_packed.shape[1]
    N = N_half * 2

    # unpack along N: low nibble = even columns, high nibble = odd columns
    lo = (W_packed & 0xF).to(torch.float32)
    hi = ((W_packed >> 4) & 0xF).to(torch.float32)

    # interleave back to (K, N)
    W_int4 = torch.zeros(K, N, dtype=torch.float32, device=W_packed.device)
    W_int4[:, 0::2] = lo
    W_int4[:, 1::2] = hi

    # dequantize per group
    W_groups = W_int4.reshape(K // group_size, group_size, N)
    W_deq = (W_groups - zeros.unsqueeze(1).float()) * scales.unsqueeze(1).float()
    return W_deq.reshape(K, N)


# ============================================================
# Reference implementations
# ============================================================


def matmul_fp16(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Standard fp16 matmul for baseline."""
    return x @ W


def matmul_int8_pytorch(
    x: torch.Tensor, W_int8: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize then matmul in PyTorch."""
    W_fp = (W_int8.float() - zero_point.float()) * scale.float()
    return x.float() @ W_fp


def matmul_int4_pytorch(
    x: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize then matmul in PyTorch."""
    W_fp = dequantize_int4(W_packed, scales, zeros, group_size)
    return x.float() @ W_fp


# ============================================================
# Triton kernels
# ============================================================


@triton.jit
def matmul_int8_kernel(
    X,
    W_int8,
    Scale,
    ZeroPoint,
    Output,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offsets_m = tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, BLOCK_K)
    offsets_n = tl.arange(0, BLOCK_N)

    offsets_row_x = pid_m * BLOCK_M + offsets_m
    mask_row_x = offsets_row_x < M

    offsets_col_w = pid_n * BLOCK_N + offsets_n
    mask_col_w = offsets_col_w < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    scale = tl.load(Scale + offsets_col_w, mask_col_w)
    zp = tl.load(ZeroPoint + offsets_col_w, mask_col_w)

    for start in tl.range(0, K, BLOCK_K):
        offsets_k_i = start + offsets_k
        mask_k_i = offsets_k_i < K

        offsets_x_i = (
            offsets_row_x[:, None] * stride_xm + offsets_k_i[None, :] * stride_xk
        )
        mask_x_i = mask_row_x[:, None] & mask_k_i[None, :]
        x = tl.load(X + offsets_x_i, mask_x_i)

        offsets_w_i = (
            offsets_k_i[:, None] * stride_wk + offsets_col_w[None, :] * stride_wn
        )
        mask_w_i = mask_k_i[:, None] & mask_col_w[None, :]

        # Triton doesn't support uint8 to fp16 directly. Annoying.
        w = (tl.load(W_int8 + offsets_w_i, mask_w_i)).to(tl.float32)
        w = (w - zp[None, :]) * scale[None, :]

        acc = tl.dot(x, w.to(tl.float16), acc)

    offsets_row_o = pid_m * BLOCK_M + offsets_m
    offsets_col_o = pid_n * BLOCK_N + offsets_n
    mask_row_o = offsets_row_o < M
    mask_col_o = offsets_col_o < N
    offsets_o = offsets_row_o[:, None] * stride_om + offsets_col_o[None, :] * stride_on
    mask_o = mask_row_o[:, None] & mask_col_o[None, :]

    tl.store(Output + offsets_o, acc.to(tl.float16), mask_o)


def matmul_int8_triton(
    x: torch.Tensor, W_int8: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    M, K = x.shape
    N = W_int8.shape[-1]
    output = torch.empty((M, N), device=x.device)

    stride_xm, stride_xk = x.stride()
    stride_wk, stride_wn = W_int8.stride()
    stride_om, stride_on = output.stride()

    BLOCK_M, BLOCK_K, BLOCK_N = 64, 32, 64
    grid = (M // BLOCK_M, N // BLOCK_N)

    matmul_int8_kernel[grid](
        x,
        W_int8,
        scale,
        zero_point,
        output,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wk,
        stride_wn,
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return output


@triton.jit
def matmul_int4_kernel(
    X,
    W_packed,
    Scales,
    Zeros,
    Output,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    stride_sk,
    stride_sn,  # scales/zeros strides
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    TODO: Implement int4 dequantized matmul (after int8 works).

    Same structure as int8 but:
    - W_packed has two int4 values per byte — need bit manipulation to unpack
    - Scales and zeros are per-group, not per-column
    - Unpack: lo = W_packed & 0xF, hi = (W_packed >> 4) & 0xF
    """
    assert GROUP_SIZE >= BLOCK_K

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offsets_m = tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, BLOCK_K)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_n_half = tl.arange(0, BLOCK_N >> 1)

    block_n_start = pid_n * BLOCK_N
    offsets_row_x = pid_m * BLOCK_M + offsets_m
    offsets_col_w = block_n_start // 2 + offsets_n_half

    w = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start in tl.range(0, K, BLOCK_K):
        offsets_k_i = start + offsets_k

        offsets_x_i = (
            offsets_row_x[:, None] * stride_xm + offsets_k_i[None, :] * stride_xk
        )
        x = tl.load(X + offsets_x_i)

        offsets_w_i = (
            offsets_k_i[:, None] * stride_wk + offsets_col_w[None, :] * stride_wn
        )

        # Triton doesn't support uint8 to fp16 directly. Annoying.
        w = tl.load(W_packed + offsets_w_i)
        w_left = (w & 0xF).to(tl.float32)
        w_right = ((w >> 4) & 0xF).to(tl.float32)
        w = tl.reshape(
            tl.join(w_left.to(tl.float16), w_right.to(tl.float16)), (BLOCK_K, BLOCK_N)
        )  # (BLOCK_K, BLOCK_N)

        block_scales = start // GROUP_SIZE
        offsets_scales = (
            block_scales * stride_sk + (block_n_start + offsets_n) * stride_sn
        )
        scales = tl.load(Scales + offsets_scales)
        zp = tl.load(Zeros + offsets_scales)

        w = (w - zp[None, :]) * scales[None, :]

        acc = tl.dot(x, w, acc)

    offsets_row_o = pid_m * BLOCK_M + offsets_m
    offsets_col_o = pid_n * BLOCK_N + offsets_n
    mask_row_o = offsets_row_o < M
    mask_col_o = offsets_col_o < N
    offsets_o = offsets_row_o[:, None] * stride_om + offsets_col_o[None, :] * stride_on
    mask_o = mask_row_o[:, None] & mask_col_o[None, :]

    tl.store(Output + offsets_o, acc.to(tl.float16), mask_o)


def matmul_int4_triton(
    x: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    M, K = x.shape
    N = W_packed.shape[-1] * 2
    output = torch.empty((M, N), device=x.device)

    stride_xm, stride_xk = x.stride()
    stride_wk, stride_wn = W_packed.stride()
    stride_om, stride_on = output.stride()
    stride_sk, stride_sn = scales.stride()

    BLOCK_M, BLOCK_K, BLOCK_N = 64, 32, 64
    grid = (M // BLOCK_M, N // BLOCK_N)

    matmul_int4_kernel[grid](
        x,
        W_packed,
        scales,
        zeros,
        output,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wk,
        stride_wn,
        stride_om,
        stride_on,
        stride_sk,
        stride_sn,
        group_size,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return output
