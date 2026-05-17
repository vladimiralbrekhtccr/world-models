#!/bin/bash
# Real multi-view 3DGS reconstruction. Sets up CUDA_HOME / LD_LIBRARY_PATH
# correctly so gsplat can JIT-compile its CUDA kernels even though the
# OHPC system doesn't ship a CUDA toolkit — we use miniconda's CUDA 12.8.
set -e
cd /scratch/vladimir_albrekht/projects/world-models
# Activate the project venv FIRST so its python comes first on PATH.
source 3DGS/.venv/bin/activate
VENV_PYTHON="$PWD/3DGS/.venv/bin/python"
# Make CUDA from miniconda visible, but don't reorder PATH (would pick up
# the conda python instead of the venv python).
export CUDA_HOME=$HOME/miniconda3
export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0"     # H100 = sm_90
# nvcc is at miniconda's bin, but we don't want it ahead of venv on PATH.
# Tell gsplat where to find it via env var.
export CUDACXX=$CUDA_HOME/bin/nvcc
echo "[run] using python: $VENV_PYTHON"
echo "[run] CUDA_HOME=$CUDA_HOME"
echo "[run] nvcc: $($CUDACXX --version | tail -1)"
CUDA_VISIBLE_DEVICES=0 "$VENV_PYTHON" MultiPano/recon_3dgs.py \
    --scene MultiPano/input/witch_hat_atelier \
    --iters 15000 --num-yaw 12 --num-pitch 3 --view-size 384 --n-init 200000 \
    --init-ply MultiPano/input/witch_hat_atelier/scene.ply \
    --ssim 0.2 \
    --out MultiPano/input/witch_hat_atelier/scene_mv2.ply
