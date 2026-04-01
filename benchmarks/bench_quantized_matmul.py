"""
Benchmark: Quantized matmul — fp16 vs int8 vs int4 (PyTorch and Triton)

Measures latency and memory for dequantize-on-the-fly matmul at various sizes.

Run on a GPU machine:
  python benchmarks/bench_quantized_matmul.py
"""

import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, ".")
from kernels.quantized_matmul import (
    quantize_int8, matmul_fp16, matmul_int8_pytorch, matmul_int8_triton,
    quantize_int4, matmul_int4_pytorch, matmul_int4_triton,
)


def fmt_ms(ms):
    return f"{ms:.4f}"


def fmt_speedup(baseline, target):
    return f"{baseline / target:.2f}x"


def weight_bytes(W):
    return W.nelement() * W.element_size()


def benchmark_latency():
    M = 128
    # (K, N) pairs — typical hidden/intermediate dims in transformers
    sizes = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (4096, 11008),  # LLaMA-7B MLP intermediate
        (8192, 8192),
    ]
    group_size = 128

    print(f"\n{'=' * 110}")
    print(f"  Latency Benchmark  (M={M}, batch of tokens × weight matrix)")
    print(f"{'=' * 110}")
    print(
        f"{'K×N':<14} {'fp16 (ms)':<12} {'int8 PT (ms)':<14} {'int8 TT (ms)':<14} "
        f"{'int4 PT (ms)':<14} {'int4 TT (ms)':<14} {'int8 TT vs fp16':<16} {'int4 TT vs fp16'}"
    )
    print("-" * 110)

    for K, N in sizes:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        W = torch.randn(K, N, device="cuda", dtype=torch.float16)

        W_int8, scale8, zp8 = quantize_int8(W)
        W_packed, scales4, zeros4 = quantize_int4(W, group_size)

        ms_fp16 = do_bench(lambda: matmul_fp16(x, W))
        ms_int8_pt = do_bench(lambda: matmul_int8_pytorch(x, W_int8, scale8, zp8))

        try:
            ms_int8_tt = do_bench(lambda: matmul_int8_triton(x, W_int8, scale8, zp8))
            int8_tt_str = fmt_ms(ms_int8_tt)
            vs_fp16_int8 = fmt_speedup(ms_fp16, ms_int8_tt)
        except Exception:
            int8_tt_str = "FAIL"
            vs_fp16_int8 = "-"

        ms_int4_pt = do_bench(lambda: matmul_int4_pytorch(x, W_packed, scales4, zeros4, group_size))

        try:
            ms_int4_tt = do_bench(lambda: matmul_int4_triton(x, W_packed, scales4, zeros4, group_size))
            int4_tt_str = fmt_ms(ms_int4_tt)
            vs_fp16_int4 = fmt_speedup(ms_fp16, ms_int4_tt)
        except Exception:
            int4_tt_str = "FAIL"
            vs_fp16_int4 = "-"

        size_label = f"{K}×{N}"
        print(
            f"{size_label:<14} {fmt_ms(ms_fp16):<12} {fmt_ms(ms_int8_pt):<14} {int8_tt_str:<14} "
            f"{fmt_ms(ms_int4_pt):<14} {int4_tt_str:<14} {vs_fp16_int8:<16} {vs_fp16_int4}"
        )


def benchmark_memory():
    sizes = [
        (4096, 4096),
        (4096, 11008),
        (8192, 8192),
    ]
    group_size = 128

    print(f"\n{'=' * 90}")
    print(f"  Weight Memory Comparison")
    print(f"{'=' * 90}")
    print(
        f"{'K×N':<14} {'fp16 (MB)':<12} {'int8 (MB)':<12} {'int4 (MB)':<12} "
        f"{'int8 saving':<14} {'int4 saving'}"
    )
    print("-" * 70)

    for K, N in sizes:
        W = torch.randn(K, N, device="cuda", dtype=torch.float16)
        W_int8, scale8, zp8 = quantize_int8(W)
        W_packed, scales4, zeros4 = quantize_int4(W, group_size)

        fp16_mb = weight_bytes(W) / 1e6
        # int8: weights + scale + zero_point
        int8_mb = (weight_bytes(W_int8) + weight_bytes(scale8) + weight_bytes(zp8)) / 1e6
        # int4: packed weights + scales + zeros
        int4_mb = (weight_bytes(W_packed) + weight_bytes(scales4) + weight_bytes(zeros4)) / 1e6

        print(
            f"{K}×{N:<8} {fp16_mb:<12.2f} {int8_mb:<12.2f} {int4_mb:<12.2f} "
            f"{fp16_mb / int8_mb:.1f}x smaller   {fp16_mb / int4_mb:.1f}x smaller"
        )


if __name__ == "__main__":
    benchmark_latency()
    benchmark_memory()
