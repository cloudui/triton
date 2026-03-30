"""
Benchmark: Batched multi-head FlashAttention — Triton vs naive vs native

Run on a GPU machine:
  python benchmarks/bench_flash_attention_full.py
"""

import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, ".")
from kernels.flash_attention_full import (
    flash_attention_full_naive,
    flash_attention_full_native,
    flash_attention_full_triton,
)


def benchmark():
    batch = 4
    n_heads = 8
    d_k = 64
    sizes = [128, 256, 512, 1024, 2048, 4096]

    for causal in [False, True]:
        label = "Causal" if causal else "Non-Causal"
        print(f"\n{'=' * 90}")
        print(f"  {label} Attention  (batch={batch}, heads={n_heads}, d_k={d_k})")
        print(f"{'=' * 90}")
        print(f"{'Seq Len':<10} {'Naive (ms)':<14} {'Flash (ms)':<14} {'Native (ms)':<14} {'Flash vs Naive':<16} {'Flash vs Native'}")
        print("-" * 80)

        for seq_len in sizes:
            Q = torch.randn(batch, n_heads, seq_len, d_k, device="cuda", dtype=torch.float16)
            K = torch.randn(batch, n_heads, seq_len, d_k, device="cuda", dtype=torch.float16)
            V = torch.randn(batch, n_heads, seq_len, d_k, device="cuda", dtype=torch.float16)

            try:
                ms_naive = do_bench(lambda: flash_attention_full_naive(Q, K, V, causal=causal))
                naive_str = f"{ms_naive:<14.4f}"
            except RuntimeError:
                ms_naive = float("inf")
                naive_str = "OOM           "

            ms_flash = do_bench(lambda: flash_attention_full_triton(Q, K, V, causal=causal))
            ms_native = do_bench(lambda: flash_attention_full_native(Q, K, V, causal=causal))

            if ms_naive == float("inf"):
                vs_naive = "OOM"
            else:
                vs_naive = f"{ms_naive / ms_flash:.2f}x"

            vs_native = f"{ms_native / ms_flash:.2f}x"

            print(f"{seq_len:<10} {naive_str} {ms_flash:<14.4f} {ms_native:<14.4f} {vs_naive:<16} {vs_native}")


if __name__ == "__main__":
    benchmark()
