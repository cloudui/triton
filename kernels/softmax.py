"""
Softmax — PyTorch reference + Triton kernel

safe softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Applied row-wise: each row is independently normalized to a probability distribution.
"""

import torch
import triton
import triton.language as tl


def softmax_pytorch(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    x = x.exp()
    return x / x.sum(dim=-1, keepdim=True)


def softmax_native(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)


@triton.jit
def softmax_kernel(
    X,  # input pointer
    Output,  # output pointer
    stride,  # row stride of X
    N,  # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + col_offsets

    mask = col_offsets < N

    # -inf otherwise defaults to 0 and exp(0) no good :(
    x = tl.load(X + offsets, mask=mask, other=float("-inf"))
    x_exp = tl.exp(x - tl.max(x))
    output = x_exp / tl.sum(x_exp)

    tl.store(Output + offsets, output, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    stride = x.shape[-1]
    N = x.shape[-1]
    BLOCK_SIZE = 1024

    output = torch.empty_like(x)

    grid = (x.shape[0],)

    softmax_kernel[grid](x, output, stride, N, BLOCK_SIZE)

    return output
