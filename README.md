# Triton GPU Kernels

Custom Triton kernels for core transformer operations — from elementwise activations to fused attention — with PyTorch reference implementations and benchmarks. Built as a learning project to understand GPU kernel programming from scratch.

## Kernels

### RMSNorm
Row-wise normalization used in Llama, Mistral, etc. Each Triton program handles one row — loads the full row, computes `sqrt(mean(x^2) + eps)`, normalizes, and scales by a learned weight. **4.5-6.4x faster than PyTorch.**

### SwiGLU
Gated activation: `SwiGLU(x, gate) = x * silu(gate)`. Elementwise operation — each Triton program processes a flat block of elements. **1.5-2.3x faster than PyTorch.**

### Softmax
Numerically stable softmax with row-wise max subtraction for fp16 precision. **4-8x faster than PyTorch, 1.2-1.8x faster than native PyTorch softmax.**

### Naive Attention
Single-head scaled dot-product attention without masking: `softmax(Q @ K^T / sqrt(d_k)) @ V`. All three operations (matmul, softmax, matmul) fused into one kernel. **1.6-2x faster than manual PyTorch implementation.**

### Fused RMSNorm + SwiGLU
Combines normalization and activation into a single kernel to avoid an extra round trip to HBM. In a standard transformer FFN, these run back-to-back, so fusing them halves global memory accesses. **5-6x faster than PyTorch, 3-4x faster than torch.compile.**

## Benchmarks

All benchmarks on NVIDIA GPU with fp16, batch size 128.

### RMSNorm: PyTorch vs Triton
```
Hidden Size     PyTorch (ms)    Native (ms)     Triton (ms)     Speedup
----------------------------------------------------------------------
1024            0.035           0.039           0.009           4.53x
2048            0.032           0.043           0.008           5.16x
4096            0.040           0.050           0.008           6.44x
8192            0.037           0.052           0.008           6.20x
```

### SwiGLU: PyTorch vs Triton
```
n=     1,024 | PyTorch: 0.0152ms | Triton: 0.0104ms | Speedup: 1.46x
n=    65,536 | PyTorch: 0.0162ms | Triton: 0.0074ms | Speedup: 2.18x
n= 1,048,576 | PyTorch: 0.0200ms | Triton: 0.0128ms | Speedup: 1.56x
n= 8,388,608 | PyTorch: 0.1015ms | Triton: 0.0438ms | Speedup: 2.32x
```

### Fused RMSNorm+SwiGLU: PyTorch vs torch.compile vs Triton
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

### Softmax: PyTorch vs Triton
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

### Naive Attention: PyTorch vs Triton (single-head, d_k=64)
```
Seq Len      PyTorch (ms)    Native (ms)     Triton (ms)    vs PyTorch   vs Native
--------------------------------------------------------------------------------
64           0.0196          0.2081          0.0098         2.01x       21.30x
128          0.0197          0.2152          0.0119         1.65x       18.04x
256          0.0204          0.2247          0.0109         1.86x       20.54x
512          0.0250          0.2278          0.0121         2.07x       18.80x
```

Note: Triton's massive speedup vs native is due to native's overhead when handling unsqueeze/squeeze on single-batch inputs. On properly batched multi-head tensors, native would be more competitive.

## Project Structure

```
kernels/
  rmsnorm.py                  # RMSNorm: row-wise normalization
  swiglu.py                   # SwiGLU: gated FFN activation
  softmax.py                  # Softmax: numerically stable row normalization
  attention.py                # Naive attention: fused scaled dot-product
  fused_rmsnorm_swiglu.py     # Fused RMSNorm + SwiGLU into one kernel
benchmarks/
  bench_rmsnorm.py            # RMSNorm performance vs PyTorch/native
  bench_swiglu.py             # SwiGLU performance vs PyTorch/native
  bench_softmax.py            # Softmax performance vs PyTorch/native
  bench_attention.py          # Naive attention performance (small seqs)
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
python benchmarks/bench_fused.py
```

## Learning Path

This repo represents a progression through Triton kernel development:

1. **RMSNorm** — Row-wise reduction + scaling. Learn grid/block programming and reductions.
2. **SwiGLU** — Elementwise operations. Learn simple parallel work distribution.
3. **Softmax** — Numerically stable reduction. Learn the max-subtract trick for fp16.
4. **Naive Attention** — 2D block loads and matrix ops. Learn `tl.dot` and multi-dimensional indexing.
5. **Fused RMSNorm+SwiGLU** — Kernel fusion. Learn why combining ops saves memory bandwidth.

Next step: **FlashAttention** — Add tiling to the naive attention kernel to handle longer sequences without materializing the full attention matrix in HBM.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA)
- Triton 2.0+
- NVIDIA GPU
