#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x);
std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out);

std::vector<paddle::Tensor> ReluCUDAForward(const paddle::Tensor& x) {
  CHECK_INPUT(x);

  return relu_cuda_forward(x);
}

std::vector<paddle::Tensor> ReluCUDABackward(const paddle::Tensor& x,
                                             const paddle::Tensor& out,
                                             const paddle::Tensor& grad_out) {
  CHECK_INPUT(x);
  CHECK_INPUT(out);
  CHECK_INPUT(grad_out);

  return relu_cuda_backward(x, out, grad_out);
}
