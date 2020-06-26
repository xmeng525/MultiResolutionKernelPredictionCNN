// gaussian_blur.cc
// This program is used for selective gaussian blur for an 3D grid.
// Author: Xiaoxu Meng

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
using namespace tensorflow;  // NOLINT(build/namespaces)

typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("KernelFilter")
    .Input("grid: float")
    .Input("kernel: float")
    .Output("output: float")
    .Doc(R"doc(
Apply gaussian blur for a 4d bilateral grid (batch, height, width, channel). The gaussian blur is 3D (height, width, depth).
)doc");

REGISTER_OP("KernelFilterGrad")
    .Input("grid: float")
    .Input("kernel: float")
    .Input("backprop: float")
    .Output("grid_grad: float")
    .Output("kernel_grad: float");

bool KernelFilterKernelLauncher(
  const GPUDevice& d,
  const float* grid,
  const float* kernel,
  const int64* grid_size,
  const int64* kernel_size,
  float* const output
);

bool KernelFilterGradKernelLauncher(
	const GPUDevice& d,
	const float* grid,
	const float* kernel,
	const float* backprop,
  const int64* grid_size,
  const int64* kernel_size,
	float* const grid_grad,
  float* const kernel_grad
);

class KernelFilterOp : public OpKernel {
 public:
  explicit KernelFilterOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& grid_tensor = context->input(0);
    // Check the input dimension.
    if (grid_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto grid = grid_tensor.flat<float>();
	
    // Grab the kernel
    const Tensor& kernel_tensor = context->input(1);
    // Check the input guide dimension.
    if (kernel_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input kernel dimension should be 4: {batch, height, width, k_x * k_y}.");
    auto kernel = kernel_tensor.flat<float>();

    auto grid_size = grid_tensor.shape().dim_sizes();
    auto kernel_size = kernel_tensor.shape().dim_sizes();
    
    // Create an output tensor
    Tensor* output_tensor = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, grid_tensor.shape(), &output_tensor));
    auto output_grid = output_tensor->template flat<float>();
   
    // Call the cuda kernel launcher
    KernelFilterKernelLauncher(
      context->eigen_device<GPUDevice>(),
      grid.data(), 
      kernel.data(),
      grid_size.data(),
      kernel_size.data(),
      output_grid.data()
    );
  }
};

class KernelFilterGradOp : public OpKernel {
 public:
  explicit KernelFilterGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& grid_tensor = context->input(0);
    // Check the input dimension.
    if (grid_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto grid = grid_tensor.flat<float>();
  
    // Grab the kernel
    const Tensor& kernel_tensor = context->input(1);
    // Check the input guide dimension.
    if (kernel_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input kernel dimension should be 4: {batch, height, width, k_x * k_y}.");
    auto kernel = kernel_tensor.flat<float>();

    // Grab the backprop gradient
    const Tensor& backprop_tensor = context->input(2);
    // Check the input guide dimension.
    if (backprop_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Backprop dimension should be 4: {batch, height, width, channel}.");
    auto backprop = backprop_tensor.flat<float>();

    auto grid_size = grid_tensor.shape().dim_sizes();
    auto kernel_size = kernel_tensor.shape().dim_sizes();
    
    // Create an output tensor
    Tensor* grid_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, grid_tensor.shape(), &grid_grad_tensor));
    auto grid_grad = grid_grad_tensor->template flat<float>();

    // Create an output tensor
    Tensor* kernel_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, kernel_tensor.shape(), &kernel_grad_tensor));
    auto kernel_grad = kernel_grad_tensor->template flat<float>();

    // Call the cuda kernel launcher
    KernelFilterGradKernelLauncher(
      context->eigen_device<GPUDevice>(),
      grid.data(), 
      kernel.data(),
      backprop.data(),
      grid_size.data(),
      kernel_size.data(),
      grid_grad.data(),
      kernel_grad.data()
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("KernelFilter").Device(DEVICE_GPU), KernelFilterOp);
REGISTER_KERNEL_BUILDER(Name("KernelFilterGrad").Device(DEVICE_GPU), KernelFilterGradOp);