#!/bin/bash
# Lyra 2.0 inference on our chapel pano_2.png with custom trajectory.
# Step 1: video diffusion → outputs/pano2/videos/pano2.mp4
# Step 2: vipe_da3_gs_recon → outputs/pano2/pano2_gs_ours/reconstructed_scene.ply
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

# Stage inputs under a sample_id-named layout (Lyra picks files by index).
STAGE="$LYRA/inputs/wm_pano2"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp "$WM/lyra_in/first_frame.png" "$STAGE/0.png"
cp "$WM/lyra_in/trajectory.npz" "$STAGE/0.npz"
cp "$WM/lyra_in/captions.json"   "$STAGE/0.json"

OUTDIR="$LYRA/outputs/pano2"
mkdir -p "$OUTDIR"

echo "=== Step 1: video diffusion (DMD, 4 steps) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m lyra_2._src.inference.lyra2_custom_traj_inference \
  --input_image_path "$STAGE" \
  --trajectory_path "$STAGE" \
  --captions_path "$STAGE" \
  --experiment lyra2 \
  --checkpoint_dir checkpoints/model \
  --output_path "$OUTDIR" \
  --num_frames 161 \
  --num_samples 1 \
  --pose_scale 1.1 \
  --resolution 480,832 \
  --use_dmd

echo "=== Step 2: 3DGS reconstruction ==="
# custom_traj writes outputs/<name>/<id>.mp4 (no `videos/` subdir).
VIDEO=$(find "$OUTDIR" -maxdepth 2 -name "*.mp4" | head -1)
echo "  video: $VIDEO"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon \
  --input_video_path "$VIDEO"

echo "=== Done ==="
find "$OUTDIR" -name "*.ply" -exec ls -la {} \;
