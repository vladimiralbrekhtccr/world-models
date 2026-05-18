#!/bin/bash
# splatfacto-big pipeline: 400-frame extract → COLMAP → splatfacto-big → PLY.
set -e

VIDEO=/scratch/vladimir_albrekht/projects/world-models/_temp/video_2026-05-17_23-43-00.mp4
PROJ=/scratch/vladimir_albrekht/projects/world-models/MultiPano/ns_room_big
PREFIX=/scratch/vladimir_albrekht/projects/world-models/MultiPano/.conda/nerfstudio

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $PREFIX
export CUDA_HOME=$PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CC=$PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$PREFIX/bin/x86_64-conda-linux-gnu-g++
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $PROJ

echo "=== Step 1: ns-process-data (400 frames) ==="
CUDA_VISIBLE_DEVICES=0 ns-process-data video \
  --data "$VIDEO" \
  --output-dir "$PROJ/processed" \
  --num-frames-target 400 \
  --camera-type perspective \
  --matching-method exhaustive \
  --feature-type sift

echo "=== Step 2: ns-train splatfacto-big ==="
CUDA_VISIBLE_DEVICES=0 ns-train splatfacto-big \
  --data "$PROJ/processed" \
  --output-dir "$PROJ/train" \
  --experiment-name room_big \
  --max-num-iterations 30000 \
  --vis tensorboard

echo "=== Step 3: export PLY ==="
CFG=$(find "$PROJ/train/room_big" -name config.yml | head -1)
CUDA_VISIBLE_DEVICES=0 python /scratch/vladimir_albrekht/projects/world-models/MultiPano/export_via_nerfstudio.py \
  --load-config "$CFG" --output-dir "$PROJ/export" --output-filename splat.ply

echo "=== DONE ==="
ls -la "$PROJ/export/splat.ply"
