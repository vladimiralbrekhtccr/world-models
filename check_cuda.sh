#!/bin/bash
export PATH=$HOME/miniconda3/bin:$PATH
export CUDA_HOME=$HOME/miniconda3
echo === nvcc ===
which nvcc; nvcc --version | tail -2
echo === cuda includes ===
ls $CUDA_HOME/include/cuda.h 2>&1
ls $CUDA_HOME/include/cuda_runtime.h 2>&1
echo === libcuda ===
ls $CUDA_HOME/lib/libcudart.so* 2>&1 | head -3
