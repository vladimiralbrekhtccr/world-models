#!/bin/bash
# One-shot installation of NVIDIA Lyra 2.0 for the MultiPano experiment.
# Lyra 2.0 is the closest published analogue to what MultiPano wants to be:
# a video-diffusion-based feed-forward 3D scene generator that takes an
# image + camera trajectory + per-chunk captions and returns explorable
# 3D Gaussians. Apache 2.0, released April 2026.
#
# Repo:    https://github.com/nv-tlabs/lyra
# Paper:   https://arxiv.org/abs/2509.19296
# Weights: https://huggingface.co/nvidia/Lyra-2.0
#
# Runtime (1× H100 80GB): ~9 min per 80 frames, or ~35s with DMD distillation.
#
# WARNING: this install takes 45-60 min end-to-end:
#   * conda env from scratch (Python 3.10)
#   * CUDA 12.8 toolkit installed inside the conda env (~3 GB)
#   * PyTorch 2.7.1 + cuDNN
#   * flash-attn built from source (10-30 min)
#   * MoGe, transformer-engine, vipe, depth_anything_3 (a few minutes each)
#   * model checkpoints from HF (~10-15 GB)
#
# Run on a node where we have GPU access (node001 or whichever is free).
# Run on the shared filesystem so the env is reachable from any node.

set -e

# ── Paths ─────────────────────────────────────────────────────────
LYRA_ROOT="/scratch/vladimir_albrekht/projects/lyra"
CONDA="$HOME/miniconda3/bin/conda"
ENV_NAME="lyra2"
ENV_PREFIX="$LYRA_ROOT/.conda/$ENV_NAME"

mkdir -p "$LYRA_ROOT" && cd "$LYRA_ROOT"

# ── 0. Clone repo (or pull) ──────────────────────────────────────
if [[ ! -d lyra ]]; then
  git clone --recursive https://github.com/nv-tlabs/lyra.git
else
  (cd lyra && git pull --recurse-submodules)
fi
cd lyra/Lyra-2

# ── 1. Conda env ─────────────────────────────────────────────────
if [[ ! -d "$ENV_PREFIX" ]]; then
  $CONDA create -y -p "$ENV_PREFIX" \
    python=3.10 pip cmake ninja libgl ffmpeg packaging \
    -c conda-forge
fi

# Activate by sourcing — the script must continue with the env active.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

CONDA_BACKUP_CXX="" $CONDA install -y -p "$ENV_PREFIX" \
  gcc=13.3.0 gxx=13.3.0 eigen zlib -c conda-forge

# ── 2. CUDA toolkit inside the env ───────────────────────────────
$CONDA install -y -p "$ENV_PREFIX" cuda -c nvidia/label/cuda-12.8.0
export CUDA_HOME=$CONDA_PREFIX

# ── 3. PyTorch ───────────────────────────────────────────────────
pip install torch==2.7.1 torchvision==0.22.1 \
  --extra-index-url https://download.pytorch.org/whl/cu128

# ── 4. Build env vars ────────────────────────────────────────────
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# ── 5. Python deps ───────────────────────────────────────────────
pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
pip install --no-build-isolation "transformer_engine[pytorch]"
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

# ── 6. flash-attn (slow — 10–30 min) ─────────────────────────────
MAX_JOBS=16 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3

# ── 7. vendored CUDA extensions ──────────────────────────────────
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

# ── 8. Model weights from HuggingFace ────────────────────────────
pip install -U huggingface_hub
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .

# ── 9. Smoke test ────────────────────────────────────────────────
PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"

echo
echo "✓ Lyra 2.0 installed at $LYRA_ROOT/lyra/Lyra-2"
echo "  conda env at: $ENV_PREFIX"
echo "  Activate with:"
echo "    source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate $ENV_PREFIX"
echo
echo "Run a preset trajectory inference (sample 4):"
echo "  cd $LYRA_ROOT/lyra/Lyra-2"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference \\"
echo "    --input_image_path assets/samples \\"
echo "    --sample_id 4 \\"
echo "    --experiment lyra2 \\"
echo "    --checkpoint_dir checkpoints/model \\"
echo "    --prompt_dir assets/samples \\"
echo "    --output_path outputs/zoomgs \\"
echo "    --num_frames_zoom_in 81 --num_frames_zoom_out 241"
