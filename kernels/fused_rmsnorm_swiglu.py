"""
Fused RMSNorm + SwiGLU — single Triton kernel

Fuses normalization and activation into one kernel to halve HBM accesses:
  Separate: load x -> normalize -> store | load normalized x -> SwiGLU -> store
  Fused:    load x -> normalize -> SwiGLU -> store
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rmsnorm_swiglu_kernel(
    X,  # input pointer
    Gate,  # SwiGLU gate pointer
    Weight,  # rmsnorm weight pointer
    Output,  # output pointer
    stride,  # row stride of X
    N,  # number of columns
    eps,  # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + col_offsets

    mask = col_offsets < N

    x = tl.load(X + offsets, mask=mask)
    weight = tl.load(Weight + col_offsets, mask=mask)
    gate = tl.load(Gate + offsets, mask=mask)

    rms = tl.sqrt(tl.sum(x * x) / stride + eps)
    x = x / rms * weight
    output = x * (gate * tl.sigmoid(gate.to(tl.float32)))

    tl.store(Output + offsets, output.to(tl.float16), mask=mask)


def fused_rmsnorm_swiglu_triton(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    N = x.shape[-1]
    stride = x.shape[-1]
    output = torch.empty_like(x)

    BLOCK_SIZE = 1024

    grid = (x.shape[0],)
    fused_rmsnorm_swiglu_kernel[grid](
        x, gate, weight, output, stride, N, eps, BLOCK_SIZE
    )

    return output
