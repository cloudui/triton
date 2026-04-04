"""
Benchmark: PyTorch vs Triton vs CUDA fused RMSNorm+SwiGLU

Run on a GPU machine (requires `make build-cuda` first):
  python benchmarks/bench_cuda_fused.py
"""

import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, ".")
from kernels.rmsnorm import rmsnorm_pytorch
from kernels.swiglu import swiglu_pytorch
from kernels.fused_rmsnorm_swiglu import fused_rmsnorm_swiglu_triton

try:
    import cuda_kernels
except ImportError:
    print("CUDA extension not found. Build it first with: make build-cuda")
    sys.exit(1)


def separate_pytorch(x, gate, weight):
    normed = rmsnorm_pytorch(x, weight)
    return swiglu_pytorch(normed, gate)


def benchmark():
    sizes = [128, 512, 1024, 2048, 4096, 8192]
    batch_size = 128

    print(f"{'Hidden':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'CUDA (ms)':<15} {'CUDA vs PyT':<14} {'CUDA vs Tri'}")
    print("-" * 80)

    for hidden_size in sizes:
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
        gate = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
        weight = torch.ones(hidden_size, device="cuda", dtype=torch.float16)

        ms_pytorch = do_bench(lambda: separate_pytorch(x, gate, weight))
        ms_triton = do_bench(lambda: fused_rmsnorm_swiglu_triton(x, gate, weight))
        ms_cuda = do_bench(lambda: cuda_kernels.fused_rmsnorm_swiglu(x, weight, gate))

        speedup_pyt = ms_pytorch / ms_cuda
        speedup_tri = ms_triton / ms_cuda

        print(
            f"{hidden_size:<10} {ms_pytorch:<15.4f} {ms_triton:<15.4f} {ms_cuda:<15.4f} {speedup_pyt:.2f}x{'':<8} {speedup_tri:.2f}x"
        )


if __name__ == "__main__":
    benchmark()
