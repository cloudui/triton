/*
 * Softmax CUDA kernel v2 — register-caching, single-pass style.
 *
 * Instead of reading from global memory three times (max, exp+sum, normalize),
 * we cache all values in registers on the first pass. This turns the remaining
 * passes into register-only work, avoiding redundant global memory reads.
 *
 * How it works:
 *   1. Each thread loads NUM_ITERS float4 vectors (8 halfs each) from its row
 *      into a register array `vals[]`, finding thread-local max along the way.
 *   2. block_reduce computes the row max across all threads (shared mem + warp
 * shuffle).
 *   3. Each thread exponentiates its cached values and sums them — all in
 * registers.
 *   4. block_reduce computes the row sum.
 *   5. Each thread normalizes its cached values and writes back to global
 * memory.
 *
 * The key optimization is that steps 3-5 never touch global memory — everything
 * lives in registers after the initial load.
 *
 * Capacity: max 256 threads × 8 values/thread × NUM_ITERS iterations.
 *   NUM_ITERS=1 → up to 2048 elements per row
 *   NUM_ITERS=2 → up to 4096
 *   NUM_ITERS=4 → up to 8192
 * Currently supports rows up to N=8192. Extending beyond that requires adding
 * more NUM_ITERS template instantiations or falling back to a multi-pass
 * approach.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

#include "reduce.cuh"

template <int NUM_ITERS>
__global__ void softmax_triton_kernel(const half *__restrict__ input,
                                      half *__restrict__ output, int N) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  extern __shared__ float shared[];

  int nvec = N / 8; // number of float4 vectors per row

  // Each thread caches NUM_ITERS * 8 values in registers — this is what
  // avoids re-reading from global memory for the exp/sum/normalize passes.
  constexpr int VALS_PER_THREAD = NUM_ITERS * 8;
  float vals[VALS_PER_THREAD];

  const float4 *row_in_vec = reinterpret_cast<const float4 *>(input + row * N);
  float4 *row_out_vec = reinterpret_cast<float4 *>(output + row * N);

  // Pass 1: Load all values into registers, find thread-local max.
  // Each float4 holds 4 half2 pairs = 8 half values.
  float thread_max = -INFINITY;
#pragma unroll
  for (int i = 0; i < NUM_ITERS; i++) {
    int idx = tid + i * blockDim.x;
    if (idx < nvec) {
      float4 chunk = row_in_vec[idx];
      half2 *h = reinterpret_cast<half2 *>(&chunk);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 f = __half22float2(h[j]);
        vals[8 * i + 2 * j] = f.x;
        vals[8 * i + 2 * j + 1] = f.y;
        thread_max = fmaxf(thread_max, fmaxf(f.x, f.y));
      }
    } else {
      // Pad with -inf so exp(-inf) = 0, contributing nothing to the sum.
#pragma unroll
      for (int j = 0; j < 8; j++) {
        vals[8 * i + j] = -INFINITY;
      }
    }
  }

  // Reduce thread-local maxes to row max (shared memory tree reduction + warp
  // shuffle).
  float row_max = block_reduce(thread_max, shared, tid, MaxOp{});

  // Pass 2 (register-only): Exponentiate and sum — no global memory access.
  float thread_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < VALS_PER_THREAD; i++) {
    vals[i] = __expf(vals[i] - row_max);
    thread_sum += vals[i];
  }

  float inv_row_sum = 1.0f / block_reduce(thread_sum, shared, tid, SumOp{});

  // Pass 3: Normalize cached values and write back to global memory.
#pragma unroll
  for (int i = 0; i < NUM_ITERS; i++) {
    int ind = tid + i * blockDim.x;
    if (ind < nvec) {
      float4 out_chunk;
      half2 *v_out = reinterpret_cast<half2 *>(&out_chunk);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 f;
        f.x = vals[8 * i + 2 * j] * inv_row_sum;
        f.y = vals[8 * i + 2 * j + 1] * inv_row_sum;
        v_out[j] = __float22half2_rn(f);
      }
      row_out_vec[ind] = out_chunk;
    }
  }
}

torch::Tensor softmax_triton_cuda(torch::Tensor input) {
  auto output = torch::empty_like(input);
  int M = input.size(0);
  int N = input.size(1);

  int threads = max(min(256, N / 8), 32);
  int blocks = M;
  int shared_mem = threads * sizeof(float);

  const half *in_ptr =
      reinterpret_cast<const half *>(input.data_ptr<at::Half>());
  half *out_ptr = reinterpret_cast<half *>(output.data_ptr<at::Half>());

  int elems_per_block = threads * 8;
  if (N <= elems_per_block) {
    softmax_triton_kernel<1>
        <<<blocks, threads, shared_mem>>>(in_ptr, out_ptr, N);
  } else if (N <= elems_per_block * 2) {
    softmax_triton_kernel<2>
        <<<blocks, threads, shared_mem>>>(in_ptr, out_ptr, N);
  } else {
    softmax_triton_kernel<4>
        <<<blocks, threads, shared_mem>>>(in_ptr, out_ptr, N);
  }

  return output;
}
