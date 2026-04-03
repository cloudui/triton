#include <torch/extension.h>

// Forward declarations — implemented in their own .cu files
torch::Tensor softmax_cuda(torch::Tensor input);
torch::Tensor softmax_triton_cuda(torch::Tensor input);
torch::Tensor rmsnorm_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &softmax_cuda, "Naive Softmax (CUDA)");
  m.def("softmax_triton", &softmax_triton_cuda, "Softmax Triton-style (CUDA)");
  m.def("rmsnorm", &rmsnorm_cuda, "RMSNorm (CUDA)");
}
