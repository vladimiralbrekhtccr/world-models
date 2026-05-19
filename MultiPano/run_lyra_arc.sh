#!/bin/bash
# Lyra-2: 481-frame orbit-AROUND-the-chapel video → recon. GPU 6.
set -e
ENV_PREFIX=/scratch/vladimir_albrekht/projects/lyra/.conda/lyra2
LYRA=/scratch/vladimir_albrekht/projects/lyra/lyra/Lyra-2
WM=/scratch/vladimir_albrekht/projects/world-models/MultiPano

cd "$LYRA"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cufft/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE="$LYRA/inputs/wm_pano2_arc"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp "$WM/lyra_in_arc/"*.png "$WM/lyra_in_arc/"*.npz "$WM/lyra_in_arc/"*.json "$STAGE/"
OUTDIR="$LYRA/outputs/pano2_arc"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

echo "=== Step 1: video diffusion (481 frames) ==="
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m lyra_2._src.inference.lyra2_custom_traj_inference \
  --input_image_path "$STAGE" --trajectory_path "$STAGE" --captions_path "$STAGE" \
  --experiment lyra2 --checkpoint_dir checkpoints/model \
  --output_path "$OUTDIR" --num_frames 481 --num_samples 1 \
  --pose_scale 1.0 --resolution 480,832 --use_dmd

echo "=== Step 2: 3DGS reconstruction (for VIPE poses) ==="
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon \
  --input_video_path "$OUTDIR/0.mp4"

echo "=== Done ==="
find "$OUTDIR" -name "*.ply" -o -name "cameras.npz" -exec ls -la {} \;
