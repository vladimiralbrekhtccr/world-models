#!/bin/bash
# End-to-end COLMAP + gsplat training on a video.
# Same logical pipeline as nerfstudio's `ns-train splatfacto`, just without
# the wrapper. Run from anywhere; needs node001 for the gsplat compile.
set -e

VIDEO="${1:-/scratch/vladimir_albrekht/projects/world-models/_temp/video_2026-05-17_22-11-25.mp4}"
WORK="${2:-/scratch/vladimir_albrekht/projects/world-models/MultiPano/colmap_room}"
OUT_PLY="${3:-/scratch/vladimir_albrekht/projects/world-models/MultiPano/colmap_room/room_gsplat.ply}"
N_FRAMES="${4:-80}"
ITERS="${5:-7000}"

ENV_PREFIX=/scratch/vladimir_albrekht/projects/lyra/.conda/lyra2
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_PREFIX
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export CUDA_HOME=$CONDA_PREFIX
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$SITE/nvidia/nvtx/include:$SITE/nvidia/cublas/include:$SITE/nvidia/cuda_runtime/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cufft/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python /scratch/vladimir_albrekht/projects/world-models/MultiPano/run_colmap_gsplat.py \
  --video "$VIDEO" --work_dir "$WORK" --out_ply "$OUT_PLY" \
  --n_frames "$N_FRAMES" --iters "$ITERS"

echo "=== DONE ==="
ls -la "$OUT_PLY"
