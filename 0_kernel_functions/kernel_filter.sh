#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o kernel_filter.cu.o kernel_filter.cu.cc \
	-I /usr/local \
  	${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG 

g++ -std=c++11 -shared -o kernel_filter.so kernel_filter.cc \
	-L /usr/local/cuda-9.0/lib64/ \
	kernel_filter.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} 

rm kernel_filter.cu.o