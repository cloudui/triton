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
    matmul_fp16,
    matmul_fp16_triton,
    matmul_int4_pytorch,
    matmul_int4_triton,
    matmul_int8_pytorch,
    matmul_int8_triton,
    quantize_int4,
    quantize_int8,
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

    # --- Latency table ---
    print(f"\n{'=' * 85}")
    print(f"  Latency  (M={M})")
    print(f"{'=' * 85}")
    print(
        f"{'K×N':<14} {'cuBLAS (ms)':<13} {'TT fp16 (ms)':<14} "
        f"{'int8 TT (ms)':<14} {'int4 TT (ms)':<14}"
    )
    print("-" * 85)

    rows = []
    for K, N in sizes:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        W = torch.randn(K, N, device="cuda", dtype=torch.float16)

        W_int8, scale8, zp8 = quantize_int8(W)
        W_packed, scales4, zeros4 = quantize_int4(W, group_size)

        ms_cublas = do_bench(lambda: matmul_fp16(x, W))

        try:
            ms_fp16_tt = do_bench(lambda: matmul_fp16_triton(x, W))
        except Exception:
            ms_fp16_tt = None

        try:
            ms_int8_tt = do_bench(lambda: matmul_int8_triton(x, W_int8, scale8, zp8))
        except Exception:
            ms_int8_tt = None

        try:
            ms_int4_tt = do_bench(
                lambda: matmul_int4_triton(x, W_packed, scales4, zeros4, group_size)
            )
        except Exception:
            ms_int4_tt = None

        rows.append((K, N, ms_cublas, ms_fp16_tt, ms_int8_tt, ms_int4_tt))

        size_label = f"{K}×{N}"
        print(
            f"{size_label:<14} {fmt_ms(ms_cublas):<13} "
            f"{fmt_ms(ms_fp16_tt) if ms_fp16_tt else 'FAIL':<14} "
            f"{fmt_ms(ms_int8_tt) if ms_int8_tt else 'FAIL':<14} "
            f"{fmt_ms(ms_int4_tt) if ms_int4_tt else 'FAIL':<14}"
        )

    # --- Speedup table ---
    print(f"\n{'=' * 100}")
    print(f"  Speedups")
    print(f"{'=' * 100}")
    print(
        f"{'K×N':<14} {'TT vs cuBLAS':<14} "
        f"{'int8 vs cuBLAS':<16} {'int4 vs cuBLAS':<16} "
        f"{'int8 vs TT fp16':<17} {'int4 vs TT fp16'}"
    )
    print("-" * 100)

    for K, N, ms_cublas, ms_fp16_tt, ms_int8_tt, ms_int4_tt in rows:
        size_label = f"{K}×{N}"
        print(
            f"{size_label:<14} "
            f"{fmt_speedup(ms_cublas, ms_fp16_tt) if ms_fp16_tt else '-':<14} "
            f"{fmt_speedup(ms_cublas, ms_int8_tt) if ms_int8_tt else '-':<16} "
            f"{fmt_speedup(ms_cublas, ms_int4_tt) if ms_int4_tt else '-':<16} "
            f"{fmt_speedup(ms_fp16_tt, ms_int8_tt) if ms_fp16_tt and ms_int8_tt else '-':<17} "
            f"{fmt_speedup(ms_fp16_tt, ms_int4_tt) if ms_fp16_tt and ms_int4_tt else '-'}"
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
        int8_mb = (
            weight_bytes(W_int8) + weight_bytes(scale8) + weight_bytes(zp8)
        ) / 1e6
        # int4: packed weights + scales + zeros
        int4_mb = (
            weight_bytes(W_packed) + weight_bytes(scales4) + weight_bytes(zeros4)
        ) / 1e6

        print(
            f"{K}×{N:<8} {fp16_mb:<12.2f} {int8_mb:<12.2f} {int4_mb:<12.2f} "
            f"{fp16_mb / int8_mb:.1f}x smaller   {fp16_mb / int4_mb:.1f}x smaller"
        )


if __name__ == "__main__":
    benchmark_latency()
    benchmark_memory()
