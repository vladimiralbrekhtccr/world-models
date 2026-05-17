#!/bin/bash
set -e
export PATH=$HOME/miniconda3/bin:$PATH
conda install -y -c "nvidia/label/cuda-12.4.0" -n base cuda-nvcc cuda-cudart-dev cuda-libraries-dev 2>&1 | tail -25
echo === nvcc ===
~/miniconda3/bin/nvcc --version
