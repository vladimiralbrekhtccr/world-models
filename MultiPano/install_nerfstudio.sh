#!/bin/bash
# Install nerfstudio in a fresh conda env on scratch (not home).
# Standard recipe: python 3.10, torch 2.1.2+cu118, colmap, tinycudann, nerfstudio
set -e

PREFIX=/scratch/vladimir_albrekht/projects/world-models/MultiPano/.conda/nerfstudio
source $HOME/miniconda3/etc/profile.d/conda.sh

echo "=== creating env at $PREFIX ==="
$HOME/miniconda3/bin/conda create -y -p "$PREFIX" python=3.10
conda activate "$PREFIX"

echo "=== installing torch 2.1.2 + cu118 ==="
pip install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

echo "=== installing cuda-toolkit 11.8 ==="
$HOME/miniconda3/bin/conda install -y -p "$PREFIX" -c "nvidia/label/cuda-11.8.0" cuda-toolkit

echo "=== installing colmap ==="
$HOME/miniconda3/bin/conda install -y -p "$PREFIX" -c conda-forge colmap

echo "=== installing tinycudann (compiles, slow ~10min) ==="
pip install --no-cache-dir ninja
pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

echo "=== installing nerfstudio ==="
pip install --no-cache-dir nerfstudio

echo "=== smoke test ==="
python -c "import nerfstudio, torch; print('nerfstudio', nerfstudio.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
which ns-process-data ns-train colmap

echo "=== DONE ==="
