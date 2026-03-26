"""
Benchmark: PyTorch vs Triton softmax

Run on a GPU machine:
  python benchmarks/bench_softmax.py
"""

import torch
from triton.testing import do_bench

import sys
sys.path.insert(0, ".")
from kernels.softmax import softmax_pytorch, softmax_native, softmax_triton


def benchmark():
    sizes = [128, 512, 1024, 2048, 4096, 8192]
    batch_size = 32

    print(f"{'Hidden':<10} {'PyTorch (ms)':<15} {'Native (ms)':<15} {'Triton (ms)':<14} {'vs PyTorch':<12} {'vs Native'}")
    print("-" * 75)

    for hidden_size in sizes:
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)

        ms_pytorch = do_bench(lambda: softmax_pytorch(x))
        ms_native = do_bench(lambda: softmax_native(x))
        ms_triton = do_bench(lambda: softmax_triton(x))

        speedup_pytorch = ms_pytorch / ms_triton
        speedup_native = ms_native / ms_triton

        print(f"{hidden_size:<10} {ms_pytorch:<15.4f} {ms_native:<15.4f} {ms_triton:<14.4f} {speedup_pytorch:.2f}x{'':<6} {speedup_native:.2f}x")


if __name__ == "__main__":
    benchmark()
