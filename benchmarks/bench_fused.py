"""
Benchmark: PyTorch vs torch.compile vs Fused Triton kernel

Run on a GPU machine:
  python benchmarks/bench_fused.py
"""

import torch
from triton.testing import do_bench

import sys
sys.path.insert(0, ".")
from kernels.rmsnorm import rmsnorm_pytorch
from kernels.swiglu import swiglu_pytorch
from kernels.fused_rmsnorm_swiglu import fused_rmsnorm_swiglu_triton


def separate_pytorch(x, gate, weight):
    normed = rmsnorm_pytorch(x, weight)
    return swiglu_pytorch(normed, gate)


compiled = torch.compile(separate_pytorch)


def benchmark():
    sizes = [128, 512, 1024, 2048, 4096, 8192]
    batch_size = 128

    print(f"{'Hidden':<10} {'PyTorch (ms)':<15} {'Compiled (ms)':<16} {'Fused (ms)':<14} {'Fused vs PyT':<14} {'Fused vs Comp'}")
    print("-" * 75)

    for hidden_size in sizes:
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
        gate = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
        weight = torch.ones(hidden_size, device="cuda", dtype=torch.float16)

        # warmup torch.compile
        for _ in range(3):
            compiled(x, gate, weight)

        ms_pytorch = do_bench(lambda: separate_pytorch(x, gate, weight))
        ms_compiled = do_bench(lambda: compiled(x, gate, weight))
        ms_fused = do_bench(lambda: fused_rmsnorm_swiglu_triton(x, gate, weight))

        speedup_pyt = ms_pytorch / ms_fused
        speedup_comp = ms_compiled / ms_fused

        print(f"{hidden_size:<10} {ms_pytorch:<15.4f} {ms_compiled:<16.4f} {ms_fused:<14.4f} {speedup_pyt:<14.2f} {speedup_comp:.2f}")


if __name__ == "__main__":
    benchmark()
