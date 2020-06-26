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

__global__ void KernelFilterKernel(
	int grid_count,
	const float* grid, 
	const float* kernel,
	const int batch,
	const int height,
	const int width,
	const int channel,
	const int k0,
	const int half_k,
	float* output
) {
	
	CUDA_1D_KERNEL_LOOP(i, grid_count)
	{	
		const int c = i % channel;
		const int w = (i / channel) % width;
		const int h = (i / (channel * width)) % height;
		const int b = (i / (channel * width * height)) % batch;

		int sx = channel;
		int sy = channel * width;
		int sb = channel * width * height;

		float out_value = 0.0f;
		float out_weight = 0.0f;

		int k_sq = k0 * k0;
		int kernel_base = k_sq * w + k_sq * width * h + k_sq * width * height * b;
		for (int ii_o = -half_k; ii_o <= half_k; ii_o++)
		{
			int xx_o = w + ii_o;
			if (xx_o < 0 || xx_o > width - 1)
				continue;
			for (int jj_o = -half_k; jj_o <= half_k; jj_o++)
			{
				int yy_o = h + jj_o;
				if (yy_o < 0 || yy_o > height - 1)
					continue;
				int kernel_idx = (ii_o + half_k) + (jj_o + half_k) * k0 + kernel_base;
				int grid_idx = c + xx_o * sx + yy_o * sy + b * sb;
				if (grid[grid_idx] > 0.0f)
				{
					out_value += grid[grid_idx] * kernel[kernel_idx];
					out_weight += kernel[kernel_idx];
				}
			}	
		}
		output[i] = out_value / (out_weight + 1e-8f);
	}
}

__global__ void KernelFilterGridGradKernel(
	int grid_count,
	const float* grid, 
	const float* kernel,
	const float* backprop,
	const int batch,
	const int height,
	const int width,
	const int channel,
	const int k0,
	const int half_k,
	float* weight_grad
) {
	
	CUDA_1D_KERNEL_LOOP(i, grid_count)
	{	
		const int c = i % channel;
		const int w = (i / channel) % width;
		const int h = (i / (channel * width)) % height;
		const int b = (i / (channel * width * height)) % batch;

		int sx = channel;
		int sy = channel * width;
		int sb = channel * width * height;

		int k_sq = k0 * k0;
		float out_value = 0.0f;
		int kernel_base = k_sq * width * height * b;
		
		for (int ii_o = -half_k; ii_o <= half_k; ii_o++)
		{
			int xx_o = w + ii_o;
			if (xx_o < 0 || xx_o > width - 1)
				continue;
			for (int jj_o = -half_k; jj_o <= half_k; jj_o++)
			{
				int yy_o = h + jj_o;
				if (yy_o < 0 || yy_o > height - 1)
					continue;
				int kernel_idx = (-ii_o + half_k) + (-jj_o + half_k) * k0 + 
					k_sq * xx_o + 
					k_sq * width * yy_o + kernel_base;
				
				float part1 = grid[i] > 0 ? kernel[kernel_idx] : 0;
				float part2 = 0;

				for (int ii_i = -half_k; ii_i <= half_k; ii_i++)
				{
					int xx_i = xx_o + ii_i;
					if (xx_i < 0 || xx_i > width - 1)
						continue;
					for(int jj_i = -half_k; jj_i <= half_k; jj_i++)
					{
						int yy_i = yy_o + jj_i;
						if (yy_i < 0 || yy_i > height - 1)
							continue;
						int grid_idx_i = c + xx_i * sx + yy_i * sy + b * sb;
						int kernel_idx_i = (ii_i + half_k) + (jj_i + half_k) * k0 +
							k_sq * xx_o + k_sq * width * yy_o + kernel_base;
						if (grid[grid_idx_i] > 0)
						{
							part2 += kernel[kernel_idx_i];
						}
					}
				}
				int grid_idx = c + xx_o * sx + yy_o * sy + b * sb;
				out_value += backprop[grid_idx] * part1 / (part2 + 1e-8f);
			}
		}
		weight_grad[i] = out_value;
	}
}
__global__ void KernelFilterKernelGradKernel(
	int weight_count,
	const float* grid, 
	const float* kernel,
	const float* backprop,
	const int batch,
	const int height,
	const int width,
	const int channel,
	const int k0,
	const int half_k,
	float* kernel_grad
) {
	
	CUDA_1D_KERNEL_LOOP(i, weight_count)
	{	
		int k_sq = k0 * k0;

		const int k = i % k_sq;
		const int w = (i / k_sq) % width;
		const int h = (i / (k_sq * width)) % height;
		const int b = (i / (k_sq * width * height)) % batch;

		int sx = channel;
		int sy = channel * width;
		int sb = channel * width * height;
		
		float out_value = 0.0f;
		int kernel_base = i - k;
		
		int xx_o = w + k % k0 - half_k;
		int yy_o = h + k / k0 - half_k;
		if (xx_o < 0 || xx_o > width - 1 || yy_o < 0 || yy_o > height - 1)
			out_value = 0.0f;
		else
		{
			for (int cc = 0; cc < channel; cc++)
			{
				float part4 = 0.0f;
				float part2 = 0.0f;

				int grid_idx = cc + sx * xx_o + sy * yy_o + sb * b;
				float part1 = grid[grid_idx];
				if (part1 > 0)
				{
					float part3 = 1.0f;
					for (int ii_i = -half_k; ii_i <= half_k; ii_i++)
					{
						int xx_i = w + ii_i;
						if (xx_i < 0 || xx_i > width - 1)
							continue;
						for (int jj_i = -half_k; jj_i <= half_k; jj_i++)
						{
							int yy_i = h + jj_i;
							if (yy_i < 0 || yy_i > height - 1)
								continue;
							int grid_idx_i = cc + sx * xx_i + sy * yy_i + sb * b;
							int kernel_idx_i = ii_i + half_k + (jj_i + half_k) * k0 + kernel_base; 
							if (grid[grid_idx_i] > 0)
							{
								part4 += grid[grid_idx_i] * kernel[kernel_idx_i];
								part2 += kernel[kernel_idx_i];
							}
						}
					}
					int grid_idx_bp = cc + sx * w + sy * h + sb * b;
					out_value += (part1 * part2 - part3 * part4) / (part2 * part2 + 1e-8f) * backprop[grid_idx_bp];
				}
			}
		}
		kernel_grad[i] = out_value;
	}
}

bool KernelFilterKernelLauncher(
	const GPUDevice& d,
	const float* grid,
	const float* kernel,
	const int64* grid_size,
	const int64* kernel_size,
	float* output
) {
	int64 batch = grid_size[0];
	int64 height = grid_size[1];
	int64 width = grid_size[2];
	int64 channel = grid_size[3];

	int64 k0 = sqrt(kernel_size[3]);
	int64 half_k = k0 / 2;

	int64 grid_count = batch * height * width * channel;
	if (grid_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(grid_count, d);
		KernelFilterKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			grid_count,
			grid,
			kernel,
			batch,
			height,
			width,
			channel,
			k0,
			half_k,
			output);
	}
	return d.ok();
}

bool KernelFilterGradKernelLauncher(
	const GPUDevice& d,
	const float* grid,
	const float* kernel,
	const float* backprop,
	const int64* grid_size,
	const int64* kernel_size,
	float* output_grad,
	float* weight_grad
) {
	int64 batch = grid_size[0];
	int64 height = grid_size[1];
	int64 width = grid_size[2];
	int64 channel = grid_size[3];

	int64 k0 = sqrt(kernel_size[3]);
	int64 half_k = k0 / 2;

	int64 grid_count = batch * height * width * channel;
	if (grid_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(grid_count, d);
		KernelFilterGridGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			grid_count,
			grid,
			kernel,
			backprop,
			batch,
			height,
			width,
			channel,
			k0,
			half_k,
			output_grad);
	}

	int64 weight_count = batch * height * width * kernel_size[3];
	if(weight_count > 0)
	{
		CudaLaunchConfig config = GetCudaLaunchConfig(weight_count, d);
		KernelFilterKernelGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			weight_count,
			grid,
			kernel,
			backprop,
			batch,
			height,
			width,
			channel,
			k0,
			half_k,
			weight_grad);
	}
	return d.ok();
}

#endif