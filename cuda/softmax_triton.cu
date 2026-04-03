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

  int nvec = N / 8;
  constexpr int VALS_PER_THREAD = NUM_ITERS * 8;
  float vals[VALS_PER_THREAD];

  const float4 *row_in_vec = reinterpret_cast<const float4 *>(input + row * N);
  float4 *row_out_vec = reinterpret_cast<float4 *>(output + row * N);

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
#pragma unroll
      for (int j = 0; j < 8; j++) {
        vals[8 * i + j] = -INFINITY;
      }
    }
  }

  float row_max = block_reduce(thread_max, shared, tid, MaxOp{});

  float thread_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < VALS_PER_THREAD; i++) {
    vals[i] = __expf(vals[i] - row_max);
    thread_sum += vals[i];
  }

  float row_sum = block_reduce(thread_sum, shared, tid, SumOp{});

#pragma unroll
  for (int i = 0; i < NUM_ITERS; i++) {
    int ind = tid + i * blockDim.x;
    if (ind < nvec) {
      float4 out_chunk;
      half2 *v_out = reinterpret_cast<half2 *>(&out_chunk);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 f;
        f.x = vals[8 * i + 2 * j] / row_sum;
        f.y = vals[8 * i + 2 * j + 1] / row_sum;
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
