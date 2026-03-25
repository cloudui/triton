"""
SwiGLU — PyTorch reference + Triton kernel

SwiGLU(x, gate) = x * silu(gate)
where silu(x) = x * sigmoid(x)

Used in Llama, Mistral, etc. as the FFN activation.
"""

import torch
import triton
import triton.language as tl


def swiglu_pytorch(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * (gate * torch.nn.functional.sigmoid(gate))


@triton.jit
def swiglu_kernel(
    X,  # input pointer
    Gate,  # gate pointer
    Output,  # output pointer
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # get thread id
    pid = tl.program_id(axis=0)

    # get block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # item by item mask, swiglu applied to each item individually
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask)
    gate = tl.load(Gate + offsets, mask=mask)

    # sigmoid only compiles on fp32
    output = x * (gate * tl.sigmoid(gate.to(tl.float32)))

    tl.store(Output + offsets, output.to(tl.float16), mask=mask)


def swiglu_native(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * torch.nn.functional.silu(gate)


def swiglu_triton(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    x_flatten = x.view(-1)
    gate_flatten = gate.view(-1)
    output = torch.empty_like(x_flatten)

    n_elements = output.numel()

    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    swiglu_kernel[grid](x_flatten, gate_flatten, output, n_elements, BLOCK_SIZE)

    return output.view(x.shape)
