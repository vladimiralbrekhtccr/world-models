#!/bin/bash
# End-to-end nerfstudio pipeline: video → COLMAP → splatfacto → PLY
set -e

VIDEO="${1:-/scratch/vladimir_albrekht/projects/world-models/_temp/video_2026-05-17_23-43-00.mp4}"
PROJECT="${2:-/scratch/vladimir_albrekht/projects/world-models/MultiPano/ns_room2}"
N_FRAMES="${3:-200}"
MAX_ITERS="${4:-30000}"

PREFIX=/scratch/vladimir_albrekht/projects/world-models/MultiPano/.conda/nerfstudio
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $PREFIX
# gsplat JIT-compiles CUDA kernels; CUDA 11.8 nvcc requires gcc<=11.
export CUDA_HOME=$PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CC=$PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$PREFIX/bin/x86_64-conda-linux-gnu-g++

mkdir -p "$PROJECT/processed"

echo "=== Step 1: ns-process-data video → COLMAP ==="
CUDA_VISIBLE_DEVICES=0 ns-process-data video \
  --data "$VIDEO" \
  --output-dir "$PROJECT/processed" \
  --num-frames-target "$N_FRAMES" \
  --camera-type perspective \
  --matching-method exhaustive \
  --feature-type sift

echo "=== Step 2: ns-train splatfacto ==="
CUDA_VISIBLE_DEVICES=0 ns-train splatfacto \
  --data "$PROJECT/processed" \
  --output-dir "$PROJECT/train" \
  --experiment-name room2 \
  --max-num-iterations "$MAX_ITERS" \
  --pipeline.model.cull_alpha_thresh 0.005 \
  --vis tensorboard

echo "=== Step 3: ns-export gaussian-splat ==="
CFG=$(find "$PROJECT/train/room2" -name "config.yml" | head -1)
ns-export gaussian-splat --load-config "$CFG" --output-dir "$PROJECT/export"

echo "=== DONE ==="
find "$PROJECT/export" -name "*.ply" -exec ls -la {} \;
