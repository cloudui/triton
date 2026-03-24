"""
SwiGLU Triton Kernel

SwiGLU(x, gate) = x * silu(gate)
where silu(x) = x * sigmoid(x)

Used in Llama, Mistral, etc. as the FFN activation.

Step 1: Implement the PyTorch reference version
Step 2: Write the Triton kernel
Step 3: Validate correctness against reference
Step 4: Benchmark both
"""

import torch
import triton
import triton.language as tl


def swiglu_pytorch(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Reference PyTorch implementation of SwiGLU.

    TODO: Implement this. It's 1-2 lines.
    - silu(gate) = gate * sigmoid(gate)   [or use torch.nn.functional.silu]
    - output = x * silu(gate)
    """
    raise NotImplementedError("Implement me first!")


@triton.jit
def swiglu_kernel(
    X,        # input pointer
    Gate,     # gate pointer
    Output,   # output pointer
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Implement the Triton kernel.

    This one is simpler than RMSNorm — it's an elementwise operation.
    - Each program handles a block of elements
    - Load x and gate, compute silu(gate), multiply, store
    """
    pass


def swiglu_triton(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that launches the Triton kernel.

    TODO: Implement after writing the kernel above.
    """
    raise NotImplementedError("Implement after writing the kernel!")
