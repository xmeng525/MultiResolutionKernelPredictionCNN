"""
description: CUDA kernel for Image Filtering and Upsampling

@author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)
"""

import os
import tensorflow as tf
from tensorflow.python.framework import ops

##################################################################

path = os.path.dirname(os.path.abspath(__file__))
path_kernel_functions = tf.resource_loader.get_path_to_datafile(
    os.path.join(path, '0_kernel_functions', 'kernel_filter.so'))

kernel_filter_lib = tf.load_op_library(path_kernel_functions)
kernel_filter = kernel_filter_lib.kernel_filter

@ops.RegisterGradient('KernelFilter')
def _kernel_filter_grad(op, grad):
	image = op.inputs[0]
	kernel = op.inputs[1]
	return kernel_filter_lib.kernel_filter_grad(image, kernel, grad)

##################################################################

path = os.path.dirname(os.path.abspath(__file__))
path_upsampling_functions = tf.resource_loader.get_path_to_datafile(
    os.path.join(path, '0_upsampling', 'upsampling.so'))

upsampling_lib = tf.load_op_library(path_upsampling_functions)
upsampling = upsampling_lib.upsampling

@ops.RegisterGradient('Upsampling')
def _upsampling_grad(op, grad):
    image = op.inputs[0]
    return upsampling_lib.upsampling_grad(image, grad)