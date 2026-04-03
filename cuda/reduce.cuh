#pragma once

#include <cuda_runtime.h>
#include <math.h>

struct MaxOp {
  __device__ float operator()(float a, float b) { return fmaxf(a, b); }
};

struct SumOp {
  __device__ float operator()(float a, float b) { return a + b; }
};

template <typename Op> __device__ float warp_reduce(float val, Op op) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = op(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename Op>
__device__ float block_reduce(float val, float *shared, int tid, Op op) {
  shared[tid] = val;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = op(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    val = warp_reduce(shared[tid], op);
    if (tid == 0)
      shared[0] = val;
  }
  __syncthreads();
  return shared[0];
}
