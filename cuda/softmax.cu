/*
 * Softmax CUDA kernel — Barebones NAIVE implementation of CUDA Softmax
 *
 * Compare to kernels/softmax.py to see what Triton abstracts away.
 *
 * Key CUDA concepts you'll encounter here:
 *   - threadIdx, blockIdx, blockDim — the thread hierarchy
 *   - __shared__ memory — fast on-chip memory shared within a block
 *   - __syncthreads() — synchronize threads within a block
 *   - Warp-level reductions — __shfl_down_sync for fast reductions
 *
 * Build with: python cuda/setup.py install
 * Or use torch.utils.cpp_extension.load() for JIT compilation
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

// Each block handles one row. Threads within the block collaborate on the
// reduction. This is equivalent to your Triton softmax where each program
// handles one row.
__global__ void softmax_kernel(const half *__restrict__ input,
                               half *__restrict__ output,
                               int N // number of columns
) {
  // Which row am I? (equivalent to tl.program_id(0) in Triton)
  int row = blockIdx.x;

  // Which thread am I within this block?
  int tid = threadIdx.x;

  // Shared memory for reductions (equivalent to tl.max / tl.sum in Triton)
  extern __shared__ float shared[];

  int nvec = N / 8;

  // Pointer to this row's data
  const float4 *row_in_vec = reinterpret_cast<const float4 *>(input + row * N);
  float4 *row_out_vec = reinterpret_cast<float4 *>(output + row * N);

  // ========================================================
  // Step 1: Find row max (for numerical stability)
  // In Triton: tl.max(x) — one line
  // In CUDA: each thread finds max of its elements, then reduce
  // ========================================================
  float thread_max = -INFINITY;
  for (int i = tid; i < nvec; i += blockDim.x) {
    float4 chunk = row_in_vec[i];
    half2 *h = reinterpret_cast<half2 *>(&chunk);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __half22float2(h[j]);
      thread_max = fmaxf(thread_max, fmaxf(f.x, f.y));
    }
  }

  // Store each thread's max to shared memory
  shared[tid] = thread_max;
  __syncthreads();

  // Tree reduction to find global max across all threads
  // This is what tl.max() does automatically in Triton
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  float row_max = shared[0];

  // ========================================================
  // Step 2: Compute exp(x - max) and sum
  // In Triton: tl.exp(x - max), tl.sum(...)
  // In CUDA: same loop + reduction pattern
  // ========================================================
  float thread_sum = 0.0f;
  for (int i = tid; i < nvec; i += blockDim.x) {
    float4 chunk = row_in_vec[i];
    half2 *v = reinterpret_cast<half2 *>(&chunk);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __half22float2(v[j]);
      thread_sum += expf(f.x - row_max) + expf(f.y - row_max);
    }
  }

  shared[tid] = thread_sum;
  __syncthreads();

  // Tree reduction for sum
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  float row_sum = shared[0];

  // ========================================================
  // Step 3: Normalize and write output
  // In Triton: output = x_exp / sum, tl.store(...)
  // In CUDA: each thread writes its elements
  // ========================================================
  for (int i = tid; i < nvec; i += blockDim.x) {
    float4 chunk = row_in_vec[i];
    float4 out_chunk;
    half2 *v = reinterpret_cast<half2 *>(&chunk);
    half2 *v_out = reinterpret_cast<half2 *>(&out_chunk);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __half22float2(v[j]);
      f.x = expf(f.x - row_max) / row_sum;
      f.y = expf(f.y - row_max) / row_sum;
      v_out[j] = __float22half2_rn(f);
    }
    row_out_vec[i] = out_chunk;
  }
}

// PyTorch binding — makes it callable from Python
torch::Tensor softmax_cuda(torch::Tensor input) {
  auto output = torch::empty_like(input);
  int M = input.size(0);
  int N = input.size(1);

  // Launch config: one block per row, 256 threads per block
  // In Triton: grid = (M,), block size is implicit
  int threads = 256;
  int blocks = M;
  int shared_mem = threads * sizeof(float);

  softmax_kernel<<<blocks, threads, shared_mem>>>(
      reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
      reinterpret_cast<half *>(output.data_ptr<at::Half>()), N);

  return output;
}
