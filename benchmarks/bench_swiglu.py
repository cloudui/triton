import sys

import torch

sys.path.insert(0, ".")

from kernels.swiglu import swiglu_pytorch, swiglu_triton
from triton.testing import do_bench

for size in [1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024]:
    x = torch.randn(size, device="cuda", dtype=torch.float16)
    gate = torch.randn(size, device="cuda", dtype=torch.float16)
    ms_pytorch = do_bench(lambda: swiglu_pytorch(x, gate))
    ms_triton = do_bench(lambda: swiglu_triton(x, gate))
    print(
        f"n={size:>10,} | PyTorch: {ms_pytorch:.4f}ms | Triton: {ms_triton:.4f}ms | Speedup: {ms_pytorch/ms_triton:.2f}x"
    )
