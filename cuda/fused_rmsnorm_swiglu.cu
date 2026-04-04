#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

#include "helpers.cuh"
#include "reduce.cuh"

template <int NUM_ITERS>
__global__ void fused_rmsnorm_swiglu_kernel(const half *__restrict__ input,
                                            half *__restrict__ output,
                                            const half *__restrict__ weight,
                                            const half *__restrict__ gate,
                                            int N // number of columns
) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  extern __shared__ float shared[];

  int nvec = N / 8;
  constexpr int VALS_PER_THREAD = NUM_ITERS * 8;
  constexpr float EPS = 0.000001;
  float vals[VALS_PER_THREAD] = {0};
  // float weights[VALS_PER_THREAD] = {0};
  // float gates[VALS_PER_THREAD] = {0};

  const float4 *row_in_vec = reinterpret_cast<const float4 *>(input + row * N);
  const float4 *weight_in = reinterpret_cast<const float4 *>(weight);
  const float4 *gate_in = reinterpret_cast<const float4 *>(gate + row * N);

  float4 *row_out_vec = reinterpret_cast<float4 *>(output + row * N);

  float thread_square_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < NUM_ITERS; i++) {
    int idx = tid + i * blockDim.x;
    if (idx < nvec) {
      float4 val_chunk = row_in_vec[idx];
      float4 weight_chunk = weight_in[idx];
      float4 gate_chunk = gate_in[idx];
      half2 *h_i = reinterpret_cast<half2 *>(&val_chunk);
      half2 *h_w = reinterpret_cast<half2 *>(&weight_chunk);
      half2 *h_g = reinterpret_cast<half2 *>(&gate_chunk);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 f_i = __half22float2(h_i[j]);
        float2 f_w = __half22float2(h_w[j]);
        float2 f_g = __half22float2(h_g[j]);
        int i1 = 8 * i + 2 * j;
        int i2 = i1 + 1;

        vals[i1] = f_i.x * f_w.x * f_g.x * sigmoid(f_g.x);
        vals[i2] = f_i.y * f_w.y * f_g.y * sigmoid(f_g.y);
        // weights[i1] = f_w.x;
        // weights[i2] = f_w.y;
        // gates[i1] = f_g.x;
        // gates[i2] = f_g.y;

        thread_square_sum += f_i.x * f_i.x + f_i.y * f_i.y;
      }
    }
  }

  float row_sum = block_reduce(thread_square_sum, shared, tid, SumOp{});
  // synced, now calculate rownorm
  float inv_rms = __frsqrt_rn(row_sum / N + EPS);

#pragma unroll
  for (int i = 0; i < NUM_ITERS; i++) {
    int ind = tid + i * blockDim.x;
    if (ind < nvec) {
      float4 out_chunk;
      half2 *v_out = reinterpret_cast<half2 *>(&out_chunk);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 f;
        f.x = vals[8 * i + 2 * j] * inv_rms;
        f.y = vals[8 * i + 2 * j + 1] * inv_rms;
        v_out[j] = __float22half2_rn(f);
      }
      row_out_vec[ind] = out_chunk;
    }
  }
}

// PyTorch binding — makes it callable from Python
torch::Tensor fused_rmsnorm_swiglu_cuda(torch::Tensor input,
                                        torch::Tensor weight,
                                        torch::Tensor gate) {
  auto output = torch::empty_like(input);
  int M = input.size(0);
  int N = input.size(1);

  int threads = max(min(256, N / 8), 32);
  int blocks = M;
  int shared_mem = threads * sizeof(float);

  const half *in_ptr =
      reinterpret_cast<const half *>(input.data_ptr<at::Half>());
  half *out_ptr = reinterpret_cast<half *>(output.data_ptr<at::Half>());
  half *weight_ptr = reinterpret_cast<half *>(weight.data_ptr<at::Half>());
  half *gate_ptr = reinterpret_cast<half *>(gate.data_ptr<at::Half>());

  int elems_per_block = threads * 8;
  if (N <= elems_per_block) {
    fused_rmsnorm_swiglu_kernel<1><<<blocks, threads, shared_mem>>>(
        in_ptr, out_ptr, weight_ptr, gate_ptr, N);
  } else if (N <= elems_per_block * 2) {
    fused_rmsnorm_swiglu_kernel<2><<<blocks, threads, shared_mem>>>(
        in_ptr, out_ptr, weight_ptr, gate_ptr, N);
  } else {
    fused_rmsnorm_swiglu_kernel<4><<<blocks, threads, shared_mem>>>(
        in_ptr, out_ptr, weight_ptr, gate_ptr, N);
  }

  return output;
}
