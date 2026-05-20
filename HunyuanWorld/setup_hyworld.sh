#!/bin/bash
# HY-World 2.0 install — everything on scratch, GPU 6 on node001.
set -e

PROJ=/scratch/vladimir_albrekht/projects/hunyuanworld
PREFIX=$PROJ/.conda/hyworld2
REPO=$PROJ/HY-World-2.0

# Redirect every cache out of $HOME
export HF_HOME=$PROJ/.cache/hf
export HF_TOKEN=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
export HF_HUB_TOKEN=$HF_TOKEN
export PIP_CACHE_DIR=$PROJ/.cache/pip
export TORCH_HOME=$PROJ/.cache/torch
export XDG_CACHE_HOME=$PROJ/.cache/xdg
export TMPDIR=$PROJ/.tmp
export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
mkdir -p $HF_HOME $PIP_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME $TMPDIR $CONDA_PKGS_DIRS

source $HOME/miniconda3/etc/profile.d/conda.sh

# ── Phase 1: env + torch 2.7.1 + cu128 ──────────────────────────────
if [ ! -d "$PREFIX" ]; then
  echo "=== Phase 1: create env (python 3.11.15) ==="
  conda create -y -p $PREFIX python=3.11.15
fi
conda activate $PREFIX
which python && python -V

if ! python -c "import torch" 2>/dev/null; then
  echo "=== Phase 1b: install torch 2.7.1 + cu128 ==="
  pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
fi

# Pin gcc 11 + nvcc inside env (same trick as nerfstudio env)
if [ ! -x "$PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]; then
  echo "=== Phase 1c: gcc 11 + nvcc 12.8 ==="
  conda install -y -c conda-forge "gcc_linux-64=11" "gxx_linux-64=11"
  conda install -y -c "nvidia/label/cuda-12.8.0" cuda-nvcc cuda-cudart-dev cuda-cccl
fi

export CUDA_HOME=$PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CC=$PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$PREFIX/bin/x86_64-conda-linux-gnu-g++
# glm headers (installed via conda-forge) — needed by gsplat_maskgaussian
export CPATH=$PREFIX/include:${CPATH:-}

# ── Phase 2: requirements.txt ───────────────────────────────────────
if ! python -c "import diffusers, transformers, peft, kornia" 2>/dev/null; then
  echo "=== Phase 2: pip install -r requirements.txt ==="
  pip install --no-cache-dir -r $REPO/requirements.txt
fi

# ── Phase 3: custom gsplat (maskgaussian) ────────────────────────────
if ! python -c "import gsplat" 2>/dev/null; then
  echo "=== Phase 3: gsplat_maskgaussian build ==="
  cd $REPO/hyworld2/worldgen/third_party/gsplat_maskgaussian
  pip install --no-cache-dir -e . --no-build-isolation
  cd $PROJ
fi

# ── Phase 4: flash-attn 2 (simpler than FA3 for now) ─────────────────
if ! python -c "import flash_attn" 2>/dev/null; then
  echo "=== Phase 4: flash-attn 2 build (~10-20 min) ==="
  MAX_JOBS=16 pip install --no-cache-dir flash-attn --no-build-isolation
fi

# ── Phase 5: smoke test ──────────────────────────────────────────────
echo "=== Phase 5: smoke test ==="
python -c "
import torch, diffusers, transformers, gsplat, flash_attn
print('OK torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'devs', torch.cuda.device_count())
print('OK diffusers', diffusers.__version__, '| transformers', transformers.__version__)
print('OK gsplat', gsplat.__version__, '| flash_attn', flash_attn.__version__)
"
echo "=== SETUP DONE ==="
