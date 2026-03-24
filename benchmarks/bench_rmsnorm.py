"""
Benchmark RMSNorm: PyTorch vs Triton

Run on a GPU machine:
  python benchmarks/bench_rmsnorm.py
"""

import torch
from triton.testing import do_bench

import sys
sys.path.insert(0, '.')
from kernels.rmsnorm import rmsnorm_pytorch, rmsnorm_native, rmsnorm_triton


def benchmark_rmsnorm():
    sizes = [1024, 2048, 4096, 8192]
    batch_size = 128

    print(f"{'Hidden Size':<15} {'PyTorch (ms)':<15} {'Native (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    for hidden_size in sizes:
        x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
        weight = torch.ones(hidden_size, device='cuda', dtype=torch.float16)

        pytorch_ms = do_bench(lambda: rmsnorm_pytorch(x, weight))
        native_ms = do_bench(lambda: rmsnorm_native(x, weight))
        triton_ms = do_bench(lambda: rmsnorm_triton(x, weight))

        speedup = native_ms / triton_ms
        print(f"{hidden_size:<15} {pytorch_ms:<15.3f} {native_ms:<15.3f} {triton_ms:<15.3f} {speedup:<10.2f}x")


if __name__ == "__main__":
    benchmark_rmsnorm()
