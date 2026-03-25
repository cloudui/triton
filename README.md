# Triton GPU Kernels

Custom Triton kernels for transformer FFN operations, with PyTorch reference implementations and benchmarks.

## Kernels

### RMSNorm
Row-wise normalization used in Llama, Mistral, etc. Each Triton program handles one row — loads the full row, computes `sqrt(mean(x^2) + eps)`, normalizes, and scales by a learned weight.

### SwiGLU
Gated activation: `SwiGLU(x, gate) = x * silu(gate)`. Elementwise operation — each Triton program processes a flat block of elements.

### Fused RMSNorm + SwiGLU
Combines both ops into a single kernel to avoid an extra round trip to HBM. In a standard transformer FFN, these run back-to-back, so fusing them halves global memory accesses.

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

## Project Structure

```
kernels/
  rmsnorm.py                  # RMSNorm: PyTorch reference + Triton kernel
  swiglu.py                   # SwiGLU: PyTorch reference + Triton kernel
  fused_rmsnorm_swiglu.py     # Fused kernel combining both ops
benchmarks/
  bench_rmsnorm.py            # RMSNorm benchmark
  bench_swiglu.py             # SwiGLU benchmark
  bench_fused.py              # Fused kernel vs PyTorch vs torch.compile
tests/
  test_kernels.py             # Correctness tests against PyTorch references
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
python benchmarks/bench_fused.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA)
- Triton 2.0+
- NVIDIA GPU
