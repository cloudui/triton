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

    print(f"{'Hidden':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'CUDA (ms)':<15} {'CUDA-v2 (ms)':<15} {'v2 vs Tri':<12} {'v2 vs v1'}")
    print("-" * 100)

    for hidden_size in sizes:
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)

        ms_pytorch = do_bench(lambda: softmax_pytorch(x))
        ms_triton = do_bench(lambda: softmax_triton(x))
        ms_cuda = do_bench(lambda: cuda_kernels.softmax(x))
        ms_cuda_v2 = do_bench(lambda: cuda_kernels.softmax_triton(x))

        speedup_triton = ms_triton / ms_cuda_v2
        speedup_v1 = ms_cuda / ms_cuda_v2

        print(
            f"{hidden_size:<10} {ms_pytorch:<15.4f} {ms_triton:<15.4f} {ms_cuda:<15.4f} {ms_cuda_v2:<15.4f} {speedup_triton:.2f}x{'':<7} {speedup_v1:.2f}x"
        )


if __name__ == "__main__":
    benchmark()
