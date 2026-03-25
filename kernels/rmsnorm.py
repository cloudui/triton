"""
RMSNorm — PyTorch reference + Triton kernel

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
"""

import torch
import triton
import triton.language as tl


def rmsnorm_pytorch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)
    x = (x / rms) * weight
    return x


def rmsnorm_native(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


@triton.jit
def rmsnorm_kernel(
    X,  # input pointer
    Weight,  # weight pointer
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

    rms = tl.sqrt(tl.sum(x * x) / stride + eps)
    output = x / rms * weight

    tl.store(Output + offsets, output, mask=mask)


def rmsnorm_triton(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    N = x.shape[-1]
    stride = x.shape[-1]
    output = torch.empty_like(x)

    BLOCK_SIZE = 1024

    grid = (x.shape[0],)
    rmsnorm_kernel[grid](x, weight, output, stride, N, eps, BLOCK_SIZE)

    return output
