"""
Benchmark: PyTorch vs Triton naive attention

Run on a GPU machine:
  python benchmarks/bench_attention.py
"""

import torch
from triton.testing import do_bench

import sys
sys.path.insert(0, ".")
from kernels.attention import attention_pytorch, attention_native, attention_triton


def benchmark():
    sizes = [64, 128, 256, 512]
    d_k = 64

    print(f"{'Seq Len':<12} {'PyTorch (ms)':<15} {'Native (ms)':<15} {'Triton (ms)':<14} {'vs PyTorch':<12} {'vs Native'}")
    print("-" * 80)

    for seq_len in sizes:
        Q = torch.randn(seq_len, d_k, device="cuda", dtype=torch.float16)
        K = torch.randn(seq_len, d_k, device="cuda", dtype=torch.float16)
        V = torch.randn(seq_len, d_k, device="cuda", dtype=torch.float16)

        ms_pytorch = do_bench(lambda: attention_pytorch(Q, K, V))
        ms_native = do_bench(lambda: attention_native(Q, K, V))
        ms_triton = do_bench(lambda: attention_triton(Q, K, V))

        speedup_pytorch = ms_pytorch / ms_triton
        speedup_native = ms_native / ms_triton

        print(f"{seq_len:<12} {ms_pytorch:<15.4f} {ms_native:<15.4f} {ms_triton:<14.4f} {speedup_pytorch:.2f}x{'':<6} {speedup_native:.2f}x")


if __name__ == "__main__":
    benchmark()
