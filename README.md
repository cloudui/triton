# Triton GPU Kernels

Custom Triton kernels for core transformer operations — from elementwise activations to FlashAttention-2 and quantized matmul — with PyTorch reference implementations and benchmarks. Built from scratch to understand GPU kernel programming, memory-aware algorithm design, and tensor core utilization.

## Kernels

### RMSNorm
Row-wise normalization used in Llama, Mistral, etc. Each Triton program handles one row — loads the full row, computes `sqrt(mean(x^2) + eps)`, normalizes, and scales by a learned weight. **4.5-6.4x faster than PyTorch.**

### SwiGLU
Gated activation: `SwiGLU(x, gate) = x * silu(gate)`. Elementwise operation — each Triton program processes a flat block of elements. **1.5-2.3x faster than PyTorch.**

### Fused RMSNorm + SwiGLU
Combines normalization and activation into a single kernel to avoid an extra round trip to HBM. In a standard transformer FFN, these run back-to-back, so fusing them halves global memory accesses. **5-6x faster than PyTorch, 3-4x faster than torch.compile.**

### Softmax
Numerically stable softmax with row-wise max subtraction for fp16 precision. **4-9x faster than PyTorch, 1.2-1.8x faster than native.**

### Naive Attention
Single-head scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`. All three operations fused into one kernel. Only works for short sequences (must fit in one block). **1.9-2.3x faster than PyTorch, 1.2-1.6x faster than native.**

### FlashAttention-2 (Full)
Batched multi-head FlashAttention-2 with optional causal masking. Tiled attention with online softmax, `tl.dot` for tensor core acceleration, causal early exit optimization, and `exp2` hardware intrinsics. O(N) memory instead of O(N²). **Achieves 84-115% of PyTorch's production FlashAttention (Tri Dao's CUDA implementation) on A100.**

### Quantized Matmul (int8 / int4)
Tiled matmul kernels that load quantized weights (int8 or int4) and dequantize on the fly to fp16 for tensor core computation. Int4 kernel handles bit-packed weights (two values per byte) with group-wise scale/zero_point. **2x / 3.8x weight memory savings.** Demonstrates the tiled matmul programming pattern (2D grid + K-loop accumulation) and quantization fundamentals.

## Benchmarks

All benchmarks on NVIDIA A100 80GB with fp16.

### RMSNorm (batch=128)
```
Hidden Size     PyTorch (ms)    Native (ms)     Triton (ms)     Speedup
----------------------------------------------------------------------
1024            0.035           0.039           0.009           4.53x
2048            0.032           0.043           0.008           5.16x
4096            0.040           0.050           0.008           6.44x
8192            0.037           0.052           0.008           6.20x
```

### SwiGLU
```
n=     1,024 | PyTorch: 0.0152ms | Triton: 0.0104ms | Speedup: 1.46x
n=    65,536 | PyTorch: 0.0162ms | Triton: 0.0074ms | Speedup: 2.18x
n= 1,048,576 | PyTorch: 0.0200ms | Triton: 0.0128ms | Speedup: 1.56x
n= 8,388,608 | PyTorch: 0.1015ms | Triton: 0.0438ms | Speedup: 2.32x
```

### Fused RMSNorm+SwiGLU (batch=128)
```
Hidden     PyTorch (ms)    Compiled (ms)    Fused (ms)     Fused vs PyT   Fused vs Comp
----------------------------------------------------------------------------------------
128        0.0376          0.0073           0.0080         4.70           0.91
512        0.0402          0.0324           0.0081         4.96           3.98
1024       0.0384          0.0272           0.0079         4.87           3.45
2048       0.0426          0.0266           0.0085         4.99           3.12
4096       0.0508          0.0274           0.0081         6.30           3.40
8192       0.0513          0.0336           0.0081         6.30           4.12
```

### Softmax (batch=32)
```
Hidden     PyTorch (ms)    Native (ms)     Triton (ms)    vs PyTorch   vs Native
---------------------------------------------------------------------------
128        0.0339          0.0099          0.0075         4.45x       1.30x
512        0.0313          0.0098          0.0071         4.36x       1.39x
1024       0.0354          0.0131          0.0071         4.98x       1.83x
2048       0.0450          0.0089          0.0071         6.30x       1.25x
4096       0.0630          0.0090          0.0076         8.32x       1.19x
8192       0.0337          0.0109          0.0077         4.39x       1.42x
```

### Naive Attention (single-head, d_k=64)
```
Seq Len      PyTorch (ms)    Native (ms)     Triton (ms)    vs PyTorch   vs Native
--------------------------------------------------------------------------------
64           0.0215          0.0152          0.0093         2.30x       1.63x
128          0.0223          0.0121          0.0102         2.18x       1.18x
256          0.0206          0.0159          0.0109         1.90x       1.46x
512          0.0254          0.0168          0.0118         2.15x       1.42x
```

### FlashAttention-2 — Non-Causal (batch=4, heads=8, d_k=64)
```
Seq Len    Naive (ms)     Flash (ms)     Native (ms)    Flash vs Naive   Flash vs Native
--------------------------------------------------------------------------------
128        0.0788         0.0124         0.0130         6.34x            1.04x
256        0.0755         0.0155         0.0170         4.88x            1.10x
512        0.0866         0.0285         0.0311         3.04x            1.09x
1024       0.3437         0.0695         0.0740         4.94x            1.06x
2048       1.7805         0.2482         0.2172         7.17x            0.88x
4096       5.6174         0.9358         0.8457         6.00x            0.90x
```

### FlashAttention-2 — Causal (batch=4, heads=8, d_k=64)
```
Seq Len    Naive (ms)     Flash (ms)     Native (ms)    Flash vs Naive   Flash vs Native
--------------------------------------------------------------------------------
128        0.2040         0.0110         0.0126         18.56x           1.15x
256        0.1470         0.0168         0.0179         8.74x            1.07x
512        0.1489         0.0306         0.0337         4.87x            1.10x
1024       0.6240         0.0604         0.0608         10.33x           1.01x
2048       3.1640         0.1730         0.1555         18.29x           0.90x
4096       10.3571        0.5845         0.4911         17.72x           0.84x
```

Native uses PyTorch's built-in `scaled_dot_product_attention` (Tri Dao's production FlashAttention CUDA implementation). Our Triton implementation achieves 84-115% of production performance at small-to-medium sequence lengths, narrowing to ~84% at seq_len=4096.

### Quantized Matmul (M=128)
```
K×N            fp16 (ms)    int8 TT (ms)   int4 TT (ms)   int8 vs fp16   int4 vs fp16   int8 mem    int4 mem
--------------------------------------------------------------------------------------------------------------
1024×1024      0.014        0.019          0.029          0.73x          0.48x          2.0x less   3.8x less
2048×2048      0.019        0.031          0.053          0.63x          0.36x          2.0x less   3.8x less
4096×4096      0.048        0.071          0.104          0.68x          0.47x          2.0x less   3.8x less
4096×11008     0.111        0.154          0.134          0.72x          0.83x          2.0x less   3.8x less
8192×8192      0.123        0.195          0.233          0.63x          0.53x          2.0x less   3.8x less
```

Quantized kernels are slower than cuBLAS fp16 because dequantization happens per-tile in fp16 before `tl.dot`. Production systems use integer tensor core instructions (int8×int8→int32) or FP8 tensor cores (H100+) to avoid this overhead. The value of quantization is memory savings (fitting larger models on fewer GPUs), not latency on individual matmuls.

## Project Structure

```
kernels/
  rmsnorm.py                  # RMSNorm: row-wise normalization
  swiglu.py                   # SwiGLU: gated FFN activation
  softmax.py                  # Softmax: numerically stable row normalization
  attention.py                # Naive attention: fused scaled dot-product
  flash_attention.py          # FlashAttention-2: single-head, tiled with online softmax
  flash_attention_full.py     # FlashAttention-2: batched, multi-head, causal
  fused_rmsnorm_swiglu.py     # Fused RMSNorm + SwiGLU into one kernel
  quantized_matmul.py         # int8/int4 quantized matmul with dequantize-on-the-fly
benchmarks/
  bench_rmsnorm.py            # RMSNorm performance
  bench_swiglu.py             # SwiGLU performance
  bench_softmax.py            # Softmax performance
  bench_attention.py          # Naive attention performance (small seqs)
  bench_flash_attention.py    # FlashAttention single-head vs naive vs native
  bench_flash_attention_full.py  # FlashAttention batched/multi-head/causal
  bench_fused.py              # Fused kernel vs PyTorch vs torch.compile
  bench_quantized_matmul.py   # Quantized matmul latency + memory savings
tests/
  test_kernels.py             # Correctness tests for all kernels
```

## Usage

```bash
# Install
pip install torch triton

# Run tests
python -m pytest tests/test_kernels.py -v

# Run benchmarks
python benchmarks/bench_rmsnorm.py
python benchmarks/bench_swiglu.py
python benchmarks/bench_softmax.py
python benchmarks/bench_attention.py
python benchmarks/bench_flash_attention.py
python benchmarks/bench_flash_attention_full.py
python benchmarks/bench_fused.py
python benchmarks/bench_quantized_matmul.py
```

## Learning Path

This repo represents a progression through Triton kernel development:

1. **SwiGLU** — Elementwise ops. Learn the Triton programming model: programs, blocks, masks.
2. **RMSNorm** — Row-wise reduction. Learn per-row reductions with `tl.sum`.
3. **Fused RMSNorm+SwiGLU** — Kernel fusion. Learn why combining ops saves HBM bandwidth.
4. **Softmax** — Numerically stable reduction. Learn the max-subtract trick for fp16.
5. **Naive Attention** — 2D block loads. Learn broadcasting and multi-dimensional indexing.
6. **FlashAttention-2 (single-head)** — Tiled attention with online softmax. Learn looping within kernels and incremental algorithms.
7. **FlashAttention-2 (full)** — Batched multi-head with causal masking. Learn 2D grids, stride-based pointer arithmetic, tensor core utilization via `tl.dot`, and causal early exit.
8. **Quantized Matmul** — int8/int4 dequantize-on-the-fly. Learn tiled matmul (2D grid + K-loop), bit packing/unpacking, group quantization, and the tradeoffs between memory savings and compute overhead.

## Testing Quantized Matmul

The quantization tests are split into four classes, separating concerns so failures are isolated:

- **`TestInt8Quantization` / `TestInt4Quantization`** — test the quantize/dequantize utilities with no Triton dependency. If the kernel crashes, these still run.
- **`TestInt8Matmul` / `TestInt4Matmul`** — test matmul correctness (PyTorch reference vs fp16 baseline, Triton vs PyTorch reference).

All tests are parametrized over multiple matrix sizes to catch dimension-dependent bugs.

### Error bounds

Tolerances are derived from quantization theory, not guesswork:

**Roundtrip error** (quantize → dequantize): each value is rounded to the nearest integer bin, so max per-element error = `scale / 2`. For `randn` weights with range ≈ 6:
- int8: `scale = 6/255 ≈ 0.024`, mean error ≈ `scale/4 ≈ 0.006`, max ≈ `scale/2 ≈ 0.012`
- int4: `scale = 6/15 ≈ 0.4`, mean error ≈ `0.1`, max ≈ `0.2`

**Matmul error** (quantized vs fp16): quantization error accumulates over the K-dimension dot product. Each output element sums K independent error terms, so the standard deviation grows as `scale * sqrt(K/12)`:
- int8, K=256: `std ≈ 0.024 * sqrt(256/12) ≈ 0.11`
- int4, K=256: `std ≈ 0.4 * sqrt(256/12) ≈ 1.85`

**Triton vs PyTorch** (same math, different execution): tight tolerance (`atol=0.1`) since the only difference is floating point accumulation order on GPU vs CPU.

### What else the tests check
- Quantized weight dtypes and value ranges (int8 in [-128, 127], int4 nibbles in [0, 15])
- Scale is positive for every column
- Memory savings are exact (int8 = 2x, int4 packed = 4x vs fp16 weight bytes)
- Output shapes match expected (M, N)

Run just the quantization utility tests (no Triton needed):
```bash
pytest tests/test_kernels.py -k "Quantization" -v
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA)
- Triton 2.0+
- NVIDIA GPU (benchmarked on A100 80GB)
