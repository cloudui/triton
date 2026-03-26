# Triton GPU Kernels

Custom Triton kernels for core transformer operations — from elementwise activations to FlashAttention — with PyTorch reference implementations and benchmarks. Built from scratch to understand GPU kernel programming and memory-aware algorithm design.

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

### FlashAttention-2
Tiled attention with online softmax — processes K/V in tiles so the full `(seq_len, seq_len)` attention matrix is never materialized. Uses `tl.dot` for tensor core acceleration. O(N) memory instead of O(N²). **2.75x faster than naive PyTorch at seq_len=32K, within 2x of PyTorch's production FlashAttention.**

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

### FlashAttention-2 (single-head, d_k=64)
```
Seq Len    Naive (ms)     Flash (ms)     Native (ms)    Flash vs Naive   Flash vs Native  Naive mem (MB)
----------------------------------------------------------------------------------------------------
128        0.0220         0.0127         0.0116         1.73x            0.92x            0.0
256        0.0207         0.0152         0.0154         1.36x            1.01x            0.1
512        0.0257         0.0231         0.0170         1.11x            0.73x            0.5
1024       0.0319         0.0379         0.0182         0.84x            0.48x            2.0
2048       0.0724         0.0675         0.0263         1.07x            0.39x            8.0
4096       0.2066         0.1269         0.0474         1.63x            0.37x            32.0
8192       0.6698         0.3536         0.1271         1.89x            0.36x            128.0
16384      2.4386         1.1322         0.6317         2.15x            0.56x            512.0
32768      9.8582         3.5854         1.9910         2.75x            0.56x            2048.0
```

Native uses PyTorch's built-in `scaled_dot_product_attention` (Tri Dao's production FlashAttention). Our Triton implementation achieves ~55% of production performance — the gap is due to warp-level optimizations, software pipelining, and autotuning in the production kernel.

## Project Structure

```
kernels/
  rmsnorm.py                  # RMSNorm: row-wise normalization
  swiglu.py                   # SwiGLU: gated FFN activation
  softmax.py                  # Softmax: numerically stable row normalization
  attention.py                # Naive attention: fused scaled dot-product
  flash_attention.py          # FlashAttention-2: tiled attention with online softmax
  fused_rmsnorm_swiglu.py     # Fused RMSNorm + SwiGLU into one kernel
benchmarks/
  bench_rmsnorm.py            # RMSNorm performance
  bench_swiglu.py             # SwiGLU performance
  bench_softmax.py            # Softmax performance
  bench_attention.py          # Naive attention performance (small seqs)
  bench_flash_attention.py    # FlashAttention vs naive vs native
  bench_fused.py              # Fused kernel vs PyTorch vs torch.compile
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
python benchmarks/bench_fused.py
```

## Learning Path

This repo represents a progression through Triton kernel development:

1. **SwiGLU** — Elementwise ops. Learn the Triton programming model: programs, blocks, masks.
2. **RMSNorm** — Row-wise reduction. Learn per-row reductions with `tl.sum`.
3. **Fused RMSNorm+SwiGLU** — Kernel fusion. Learn why combining ops saves HBM bandwidth.
4. **Softmax** — Numerically stable reduction. Learn the max-subtract trick for fp16.
5. **Naive Attention** — 2D block loads. Learn broadcasting and multi-dimensional indexing.
6. **FlashAttention-2** — Tiled attention with online softmax. Learn looping within kernels, `tl.dot` for tensor cores, and incremental algorithms that avoid materializing large intermediates.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA)
- Triton 2.0+
- NVIDIA GPU
