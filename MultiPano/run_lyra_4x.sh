#!/bin/bash
# Run Lyra 2.0 on 4 first-frames (N/E/S/W slices of pano_2) in ONE process,
# then 4 separate recon steps. Total runtime on cold H100: ~5 min model load
# + 4 × ~80 s sampling + 4 × ~30 s recon ≈ 12 min.
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

STAGE="$LYRA/inputs/wm_pano2_4x"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp "$WM/lyra_in_4x/"*.png "$WM/lyra_in_4x/"*.npz "$WM/lyra_in_4x/"*.json "$STAGE/"

OUTDIR="$LYRA/outputs/pano2_4x"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

echo "=== Step 1: video diffusion x4 (DMD) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m lyra_2._src.inference.lyra2_custom_traj_inference \
  --input_image_path "$STAGE" \
  --trajectory_path "$STAGE" \
  --captions_path "$STAGE" \
  --experiment lyra2 \
  --checkpoint_dir checkpoints/model \
  --output_path "$OUTDIR" \
  --num_frames 161 \
  --num_samples 4 \
  --pose_scale 1.1 \
  --resolution 480,832 \
  --use_dmd

echo "=== Step 2: 3DGS reconstruction x4 ==="
for i in 0 1 2 3; do
  V="$OUTDIR/$i.mp4"
  if [ ! -f "$V" ]; then echo "missing $V"; continue; fi
  echo "  recon id=$i: $V"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon \
    --input_video_path "$V"
done

echo "=== Done ==="
find "$OUTDIR" -name "*.ply" -exec ls -la {} \;
