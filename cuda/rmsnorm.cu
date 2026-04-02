#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

// Each block handles one row. Threads within the block collaborate on the
// reduction. This is equivalent to your Triton softmax where each program
// handles one row.
__global__ void rmsnorm_kernel(const half *__restrict__ input,
                               half *__restrict__ output,
                               int N // number of columns
) {
  // Which row am I? (equivalent to tl.program_id(0) in Triton)
  int row = blockIdx.x;

  // Which thread am I within this block?
  int tid = threadIdx.x;

  // Shared memory for reductions (equivalent to tl.max / tl.sum in Triton)
  extern __shared__ float shared[];

  // Pointer to this row's data
  const half *row_in = input + row * N;
  half *row_out = output + row * N;

  // ========================================================
  // Step 1: Find row max (for numerical stability)
  // In Triton: tl.max(x) — one line
  // In CUDA: each thread finds max of its elements, then reduce
  // ========================================================
  float thread_max = -INFINITY;
  for (int i = tid; i < N; i += blockDim.x) {
    thread_max = fmaxf(thread_max, __half2float(row_in[i]));
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
  for (int i = tid; i < N; i += blockDim.x) {
    thread_sum += expf(__half2float(row_in[i]) - row_max);
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
  for (int i = tid; i < N; i += blockDim.x) {
    float val = expf(__half2float(row_in[i]) - row_max) / row_sum;
    row_out[i] = __float2half(val);
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

// Register as a PyTorch extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &softmax_cuda, "Softmax (CUDA)");
}
