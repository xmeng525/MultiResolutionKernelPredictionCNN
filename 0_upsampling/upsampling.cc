// gaussian_blur.cc
// This program is used for selective gaussian blur for an 3D grid.
// Author: Xiaoxu Meng

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
using namespace tensorflow;  // NOLINT(build/namespaces)

typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("Upsampling")
    .Input("input: float")
    .Output("output: float")
    .Doc(R"doc(
Apply upsampling for a 4d image (batch, height, width, channel).
)doc");

REGISTER_OP("UpsamplingGrad")
    .Input("grid: float")
    .Input("backprop: float")
    .Output("grid_grad: float");

bool UpsamplingKernelLauncher(
  const GPUDevice& d,
  const float* input_image,
  const int64* input_shape,
  float* const output_image
);

bool UpsamplingGradKernelLauncher(
	const GPUDevice& d,
	const float* input_image,
	const float* backprop,
  const int64* input_shape,
  float* const grad
);

class UpsamplingOp : public OpKernel {
 public:
  explicit UpsamplingOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    // Check the input dimension.
    if (input_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto input_image = input_tensor.flat<float>();
    auto input_shape = input_tensor.shape().dim_sizes();
    
    // Create an output tensor
    auto output_shape = input_tensor.shape();
    output_shape.set_dim(1, input_tensor.shape().dim_size(1) * 2);
    output_shape.set_dim(2, input_tensor.shape().dim_size(2) * 2);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output_image = output_tensor->template flat<float>();
   
    // Call the cuda kernel launcher
    UpsamplingKernelLauncher(
      context->eigen_device<GPUDevice>(),
      input_image.data(), 
      input_shape.data(),
      output_image.data()
    );
  }
};

class UpsamplingGradOp : public OpKernel {
 public:
  explicit UpsamplingGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    // Check the input dimension.
    if (input_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto input_image = input_tensor.flat<float>();

    // Grab the backprop gradient
    const Tensor& backprop_tensor = context->input(1);
    // Check the input guide dimension.
    if (backprop_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Backprop dimension should be 4: {batch, height, width, channel}.");
    auto backprop = backprop_tensor.flat<float>();

    auto input_shape = input_tensor.shape().dim_sizes();

    // Create an output tensor
    Tensor* grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &grad_tensor));
    auto grad = grad_tensor->template flat<float>();

    // Call the cuda kernel launcher
    UpsamplingGradKernelLauncher(
      context->eigen_device<GPUDevice>(),
      input_image.data(),
      backprop.data(),
      input_shape.data(),
      grad.data()
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("Upsampling").Device(DEVICE_GPU), UpsamplingOp);
REGISTER_KERNEL_BUILDER(Name("UpsamplingGrad").Device(DEVICE_GPU), UpsamplingGradOp);