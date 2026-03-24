"""
Fused RMSNorm + SwiGLU Triton Kernel

This is the payoff — fusing two ops into one kernel to avoid
extra global memory reads/writes.

In a normal transformer FFN:
  1. RMSNorm the input        (read x from HBM, write normalized x to HBM)
  2. SwiGLU activation        (read normalized x from HBM, write output to HBM)

Fused version:
  1. Read x once, normalize in SRAM, apply SwiGLU, write output once

That's 2x fewer HBM accesses — which is the whole point of kernel fusion.

TODO: Implement this after you've got both individual kernels working.
This is where you'll really feel the performance difference.
"""

import torch
import triton
import triton.language as tl

# TODO: Write the fused kernel and wrapper function
