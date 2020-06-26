#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o upsampling.cu.o upsampling.cu.cc \
	-I /usr/local \
  	${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG 

g++ -std=c++11 -shared -o upsampling.so upsampling.cc \
	-L /usr/local/cuda-10.0/lib64/ \
	upsampling.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} 

rm upsampling.cu.o