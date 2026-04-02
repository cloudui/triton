"""
Benchmark: PyTorch vs Triton vs CUDA softmax

Run on a GPU machine (requires `make build-cuda` first):
  python benchmarks/bench_cuda_softmax.py
"""

import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, ".")
from kernels.softmax import softmax_native, softmax_pytorch, softmax_triton

try:
    import cuda_kernels
except ImportError:
    print("CUDA extension not found. Build it first with: make build-cuda")
    sys.exit(1)


def benchmark():
    sizes = [128, 512, 1024, 2048, 4096, 8192]
    batch_size = 32

    print(f"{'Hidden':<10} {'PyTorch (ms)':<15} {'Native (ms)':<15} {'Triton (ms)':<15} {'CUDA (ms)':<15} {'CUDA vs Py':<12} {'CUDA vs Tri'}")
    print("-" * 100)

    for hidden_size in sizes:
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)

        ms_pytorch = do_bench(lambda: softmax_pytorch(x))
        ms_native = do_bench(lambda: softmax_native(x))
        ms_triton = do_bench(lambda: softmax_triton(x))
        ms_cuda = do_bench(lambda: cuda_kernels.softmax(x))

        speedup_pytorch = ms_pytorch / ms_cuda
        speedup_triton = ms_triton / ms_cuda

        print(
            f"{hidden_size:<10} {ms_pytorch:<15.4f} {ms_native:<15.4f} {ms_triton:<15.4f} {ms_cuda:<15.4f} {speedup_pytorch:.2f}x{'':<7} {speedup_triton:.2f}x"
        )


if __name__ == "__main__":
    benchmark()
