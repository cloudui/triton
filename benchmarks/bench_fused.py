"""
Benchmark: Separate RMSNorm+SwiGLU vs Fused kernel

This is the final benchmark that shows why kernel fusion matters.

Run on a GPU machine:
  python benchmarks/bench_fused.py
"""

import torch
from triton.testing import do_bench

import sys
sys.path.insert(0, '.')
from kernels.rmsnorm import rmsnorm_pytorch, rmsnorm_triton
from kernels.swiglu import swiglu_pytorch, swiglu_triton
# from kernels.fused_rmsnorm_swiglu import fused_rmsnorm_swiglu_triton  # uncomment when ready


def benchmark_fused():
    sizes = [1024, 2048, 4096, 8192]
    batch_size = 128

    print(f"{'Hidden Size':<15} {'Separate PyT (ms)':<20} {'Separate Tri (ms)':<20} {'Fused Tri (ms)':<18} {'Fusion Speedup':<15}")
    print("-" * 88)

    for hidden_size in sizes:
        x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
        gate = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
        weight = torch.ones(hidden_size, device='cuda', dtype=torch.float16)

        # Separate PyTorch
        def separate_pytorch():
            normed = rmsnorm_pytorch(x, weight)
            return swiglu_pytorch(normed, gate)
        pytorch_ms = do_bench(separate_pytorch)

        # Separate Triton
        def separate_triton():
            normed = rmsnorm_triton(x, weight)
            return swiglu_triton(normed, gate)
        triton_sep_ms = do_bench(separate_triton)

        # Fused Triton (uncomment when ready)
        # fused_ms = do_bench(lambda: fused_rmsnorm_swiglu_triton(x, gate, weight))
        fused_ms = float('nan')

        fusion_speedup = triton_sep_ms / fused_ms if fused_ms == fused_ms else "N/A"
        print(f"{hidden_size:<15} {pytorch_ms:<20.3f} {triton_sep_ms:<20.3f} {fused_ms:<18.3f} {fusion_speedup:<15}")


if __name__ == "__main__":
    benchmark_fused()
