"""
RMSNorm Triton Kernel

RMSNorm(x) = x * (1 / sqrt(mean(x^2) + eps)) * weight

Step 1: Implement the PyTorch reference version
Step 2: Write the Triton kernel
Step 3: Validate correctness against reference
Step 4: Benchmark both
"""

import torch
import triton
import triton.language as tl


def rmsnorm_pytorch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Reference PyTorch implementation of RMSNorm.

    TODO: Implement this first. It's ~3 lines of math.
    - Compute the root mean square: rms = sqrt(mean(x^2) + eps)
    - Normalize: x_norm = x / rms
    - Scale: output = x_norm * weight
    """
    rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)
    x = (x / rms) * weight
    return x


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
    """
    TODO: Implement the Triton kernel after you've got the PyTorch version working.

    Hints:
    - Each program instance handles one row of the input
    - Use tl.program_id(0) to get the row index
    - Load a row, compute RMS, normalize, multiply by weight, store
    """
    pass


def rmsnorm_native(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    print(x.shape)
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def rmsnorm_triton(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Wrapper that launches the Triton kernel.

    TODO: Implement after writing the kernel above.
    - Allocate output tensor
    - Compute grid size (one program per row)
    - Launch rmsnorm_kernel
    """
    raise NotImplementedError("Implement after writing the kernel!")
