#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

template <typename data_t>
void relu_cpu_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             int64_t x_numel) {
  for (int64_t i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<data_t>(0.), x_data[i]);
  }
}

template <typename data_t>
void relu_cpu_backward_kernel(const data_t* grad_out_data,
                              const data_t* out_data,
                              data_t* grad_x_data,
                              int64_t out_numel) {
  for (int64_t i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> ReluCPUForward(const paddle::Tensor& x) {
  CHECK_INPUT(x);

  auto out = paddle::empty_like(x);

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cpu_forward_kernel", ([&] {
        relu_cpu_forward_kernel<data_t>(
            x.data<data_t>(), out.data<data_t>(), x.numel());
      }));

  return {out};
}

std::vector<paddle::Tensor> ReluCPUBackward(const paddle::Tensor& x,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out) {
  CHECK_INPUT(x);
  CHECK_INPUT(out);
  CHECK_INPUT(grad_out);

  auto grad_x = paddle::empty_like(x);

  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cpu_backward_kernel", ([&] {
                               relu_cpu_backward_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   out.data<data_t>(),
                                   grad_x.data<data_t>(),
                                   out.numel());
                             }));

  return {grad_x};
}
