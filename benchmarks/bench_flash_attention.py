"""
Benchmark: Naive PyTorch vs FlashAttention Triton vs PyTorch native (sdpa)

Run on a GPU machine:
  python benchmarks/bench_flash_attention.py
"""

import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, ".")
from kernels.flash_attention import flash_attention_naive, flash_attention_triton


def native_attention(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(
        Q.unsqueeze(0).unsqueeze(0),
        K.unsqueeze(0).unsqueeze(0),
        V.unsqueeze(0).unsqueeze(0),
    ).squeeze(0).squeeze(0)


def benchmark():
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    d_k = 64

    print(f"{'Seq Len':<10} {'Naive (ms)':<14} {'Flash (ms)':<14} {'Native (ms)':<14} {'Flash vs Naive':<16} {'Flash vs Native':<16} {'Naive mem (MB)'}")
    print("-" * 100)

    for seq_len in sizes:
        Q = torch.randn(seq_len, d_k, device="cuda", dtype=torch.float16)
        K = torch.randn(seq_len, d_k, device="cuda", dtype=torch.float16)
        V = torch.randn(seq_len, d_k, device="cuda", dtype=torch.float16)

        naive_mem_mb = seq_len * seq_len * 2 / (1024 * 1024)

        try:
            ms_naive = do_bench(lambda: flash_attention_naive(Q, K, V))
            naive_str = f"{ms_naive:<14.4f}"
        except RuntimeError:
            ms_naive = float("inf")
            naive_str = "OOM           "

        ms_flash = do_bench(lambda: flash_attention_triton(Q, K, V))
        ms_native = do_bench(lambda: native_attention(Q, K, V))

        if ms_naive == float("inf"):
            vs_naive_str = "OOM"
        else:
            vs_naive_str = f"{ms_naive / ms_flash:.2f}x"

        vs_native_str = f"{ms_native / ms_flash:.2f}x"

        print(f"{seq_len:<10} {naive_str} {ms_flash:<14.4f} {ms_native:<14.4f} {vs_naive_str:<16} {vs_native_str:<16} {naive_mem_mb:.1f}")


if __name__ == "__main__":
    benchmark()
