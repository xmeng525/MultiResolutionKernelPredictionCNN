// gaussian_blur.cu.cc
// This program is used for selective gaussian blur for an 3D grid.
// Author: Xiaoxu Meng

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <iostream>

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__device__ float get_corner(int curr_w, int curr_h, 
	int out_width, int out_height) {
	float curr_weight = 0.0f;
	if (curr_w - 1 >= 0)
	{
		if(curr_h - 1 >= 0)
		{
			curr_weight += 1.0f;
		}
		if(curr_h + 1 < out_height)
		{
			curr_weight += 1.0f;
		}
	}
	if(curr_w + 1 < out_width)
	{
		if(curr_h - 1 >= 0)
		{
			curr_weight += 1.0f;
		}
		if(curr_h + 1 < out_height)
		{
			curr_weight += 1.0f;
		}
	}
	return curr_weight;
}

__device__ float get_edge(int curr_w, int out_width) {
	float curr_weight = 0.0f;
	if(curr_w - 1 >= 0)
	{
		curr_weight += 1.0f;
	}
	if(curr_w + 1 < out_width)
	{
		curr_weight += 1.0f;
	}
	return curr_weight;
}

__global__ void UpsamplingKernel(
	int output_count,
	const float* input_image, 
	const int batch,
	const int height,
	const int width,
	const int channel,
	float* output_image
) {
	
	CUDA_1D_KERNEL_LOOP(i, output_count)
	{	
		const int c = i % channel;
		const int w = (i / channel) % width;
		const int h = (i / (channel * width)) % height;
		const int b = (i / (channel * width * height)) % batch;

		const int sx = channel;
		const int sy = channel * width/2;
		const int sb = channel * width/2 * height/2;

		float out_value = 0.0f;
		float out_weight = 0.0f;
		
		const int x0 = floor(w / 2.0f + 0.05f);
		const int x1 = ceil(w / 2.0f - 0.05f);
		const int y0 = floor(h / 2.0f + 0.05f);
		const int y1 = ceil(h / 2.0f - 0.05f);

		const int coord0 = c + x0 * sx + y0 * sy + b * sb;
		const int coord1 = c + x1 * sx + y0 * sy + b * sb;
		const int coord2 = c + x0 * sx + y1 * sy + b * sb;
		const int coord3 = c + x1 * sx + y1 * sy + b * sb;

		if (x0 >= 0 && x0 < width/2)
		{
			if (y0 >=0 && y0 < height/2)
			{
				out_value += input_image[coord0];
				out_weight += 1.0f;
			}
			if (y1 >=0 && y1 < height/2)
			{
				out_value += input_image[coord2];
				out_weight += 1.0f;
			}
		}
		if (x1 >= 0 && x1 < width/2)
		{
			if (y0 >=0 && y0 < height/2)
			{
				out_value += input_image[coord1];
				out_weight += 1.0f;
			}
			if (y1 >=0 && y1 < height/2)
			{
				out_value += input_image[coord3];
				out_weight += 1.0f;
			}
		}
		output_image[i] = out_value / (out_weight + 1e-8f);
	}
}

__global__ void UpsamplingGradKernel(
	int input_count,
	const float* input_image, 
	const float* backprop,
	const int batch,
	const int height,
	const int width,
	const int channel,
	float* grad
) {
	
	CUDA_1D_KERNEL_LOOP(i, input_count)
	{	
		const int c = i % channel;
		const int w = (i / channel) % width;
		const int h = (i / (channel * width)) % height;
		const int b = (i / (channel * width * height)) % batch;

		int out_w = w * 2;
		int out_h = h * 2;
		int out_width = width * 2;
		int out_height = height * 2;

		const int sx = channel;
		const int sy = channel * width * 2;
		const int sb = channel * width * height * 4;

		float grad_value = 0.0f;
		
		int curr_w;
		int curr_h;
		int idx_bp;
		// out_w - 1, out_h - 1
		curr_w = out_w - 1;
		curr_h = out_h - 1;
		if (curr_w < out_width && curr_w >= 0 && curr_h < out_height && curr_h >= 0)
		{
			idx_bp = c + curr_w * sx + curr_h * sy + b * sb;
			grad_value += 1.0f / get_corner(curr_w, curr_h, out_width, out_height) * backprop[idx_bp];
		}

		// out_w - 1, out_h + 1
		curr_w = out_w - 1;
		curr_h = out_h + 1;
		if (curr_w < out_width && curr_w >= 0 && curr_h < out_height && curr_h >= 0)
		{
			idx_bp = c + curr_w * sx + curr_h * sy + b * sb;
			grad_value += 1.0f / get_corner(curr_w, curr_h, out_width, out_height) * backprop[idx_bp];
		}

		// out_w + 1, out_h - 1
		curr_w = out_w + 1;
		curr_h = out_h - 1;
		if (curr_w < out_width && curr_w >= 0 && curr_h < out_height && curr_h >= 0)
		{
			idx_bp = c + curr_w * sx + curr_h * sy + b * sb;
			grad_value += 1.0f / get_corner(curr_w, curr_h, out_width, out_height) * backprop[idx_bp];
		}

		// out_w + 1, out_h + 1
		curr_w = out_w + 1;
		curr_h = out_h + 1;
		if (curr_w < out_width && curr_w >= 0 && curr_h < out_height && curr_h >= 0)
		{
			idx_bp = c + curr_w * sx + curr_h * sy + b * sb;
			grad_value += 1.0f / get_corner(curr_w, curr_h, out_width, out_height) * backprop[idx_bp];
		}
		
		// out_w, out_h - 1
		curr_h = out_h - 1;
		if (curr_h < out_height && curr_h >= 0)
		{
			idx_bp = c + out_w * sx + curr_h * sy + b * sb;
			grad_value += 1.0f / get_edge(curr_h, out_height) * backprop[idx_bp];
		}		

		// out_w, out_h + 1
		curr_h = out_h + 1;
		if (curr_h < out_height && curr_h >= 0)
		{
			idx_bp = c + out_w * sx + curr_h * sy + b * sb;
			grad_value += 1.0f / get_edge(curr_h, out_height) * backprop[idx_bp];
		}	

		// out_w - 1, out_h
		curr_w = out_w - 1;
		if (curr_w < out_width && curr_w >= 0)
		{
			idx_bp = c + curr_w * sx + out_h * sy + b * sb;
			grad_value += 1.0f / get_edge(curr_w, out_width) * backprop[idx_bp];
		}	

		// out_w + 1, out_h
		curr_w = out_w + 1;
		if (curr_w < out_width && curr_w >= 0)
		{
			idx_bp = c + curr_w * sx + out_h * sy + b * sb;
			grad_value += 1.0f / get_edge(curr_w, out_width) * backprop[idx_bp];
		}		

		idx_bp = c + out_w * sx + out_h * sy + b * sb;
		grad_value += backprop[idx_bp];
		grad[i] = grad_value;
	}
}

bool UpsamplingKernelLauncher(
	const GPUDevice& d,
	const float* input_image,
	const int64* input_shape,
	float* const output_image
) {
	int64 batch = input_shape[0];
	int64 height = input_shape[1] * 2;
	int64 width = input_shape[2] * 2;
	int64 channel = input_shape[3];

	int64 output_count = batch * height * width * channel;
	if (output_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(output_count, d);
		UpsamplingKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			output_count,
			input_image,
			batch,
			height,
			width,
			channel,
			output_image);
	}
	return d.ok();
}

bool UpsamplingGradKernelLauncher(
	const GPUDevice& d,
	const float* input_image,
	const float* backprop,
	const int64* input_shape,
	float* const grad
) {
	int64 batch = input_shape[0];
	int64 height = input_shape[1];
	int64 width = input_shape[2];
	int64 channel = input_shape[3];

	int64 input_count = batch * height * width * channel;
	if (input_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(input_count, d);
		UpsamplingGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			input_count,
			input_image,
			backprop,
			batch,
			height,
			width,
			channel,
			grad);
	}
	return d.ok();
}

#endif