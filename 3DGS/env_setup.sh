#!/usr/bin/env bash
# One-time uv venv for the 3DGS experiment. Run from inside 3DGS/.
# Idempotent: re-running upgrades pinned packages but does not recreate the venv.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# .venv inside 3DGS/ — gitignored
uv venv --python 3.11 --seed

# PyTorch with CUDA 12.4 wheels (H100-compatible)
uv pip install \
  "torch==2.4.0" "torchvision==0.19.0" \
  --index-url https://download.pytorch.org/whl/cu124

# Project deps
uv pip install \
  "numpy<2" pillow tqdm imageio[ffmpeg] plyfile opencv-python-headless \
  transformers huggingface_hub safetensors accelerate \
  diffusers timm \
  gsplat

echo
echo "[setup] done."
echo "        activate with: source 3DGS/.venv/bin/activate"
echo "        cuda check:    python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'"
