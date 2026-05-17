#!/bin/bash
set -e
cd /scratch/vladimir_albrekht/projects/world-models
source 3DGS/.venv/bin/activate
VENV_PYTHON="$PWD/3DGS/.venv/bin/python"
export CUDA_HOME=$HOME/miniconda3
export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export CUDACXX=$CUDA_HOME/bin/nvcc
export TORCH_CUDA_ARCH_LIST="9.0"
CUDA_VISIBLE_DEVICES=1 "$VENV_PYTHON" MultiPano/refine_3dgs_sds.py \
    --in-ply  MultiPano/input/witch_hat_atelier/scene.ply \
    --out-ply MultiPano/input/witch_hat_atelier/scene_sds.ply \
    --n-novel-views 24 --n-iters 1500 --view-size 384 --strength 0.35
