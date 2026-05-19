#!/bin/bash
# Resume the 4K plant pipeline — COLMAP already finished (sparse/0 exists),
# the process died before writing transforms.json. Finalize → train → export.
set -e

PROJ=/scratch/vladimir_albrekht/projects/world-models/MultiPano/ns_plant_4k
VIDEO=/scratch/vladimir_albrekht/projects/world-models/_temp/videos/plant_4K_VID20260519141238.mp4
PREFIX=/scratch/vladimir_albrekht/projects/world-models/MultiPano/.conda/nerfstudio
KSPLAT=/scratch/vladimir_albrekht/projects/world-models/MultiPano/ksplat
EXP=plant_4k

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $PREFIX
export CUDA_HOME=$PREFIX PATH=$PREFIX/bin:$PATH LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export CC=$PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$PREFIX/bin/x86_64-conda-linux-gnu-g++
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6

echo "=== Step 1b: finalize ns-process-data (skip COLMAP — already done) ==="
ns-process-data video \
  --data "$VIDEO" \
  --output-dir "$PROJ/processed" \
  --num-frames-target 400 \
  --camera-type perspective \
  --skip-colmap --skip-image-processing \
  --colmap-model-path "$PROJ/processed/colmap/sparse/0"

echo "=== Step 2: ns-train splatfacto-big ==="
ns-train splatfacto-big \
  --data "$PROJ/processed" \
  --output-dir "$PROJ/train" \
  --experiment-name "$EXP" \
  --max-num-iterations 30000 \
  --vis tensorboard

echo "=== Step 3: export PLY ==="
CFG=$(find "$PROJ/train/$EXP" -name config.yml | head -1)
python /scratch/vladimir_albrekht/projects/world-models/MultiPano/export_via_nerfstudio.py \
  --load-config "$CFG" --output-dir "$PROJ/export" --output-filename splat.ply

echo "=== Step 4: PLY → ksplat ==="
$HOME/miniconda3/envs/node20/bin/node $KSPLAT/convert.mjs \
  "$PROJ/export/splat.ply" "$PROJ/export/scene.ksplat" 1 1 2

echo "=== Step 5: flythrough video ==="
ns-render interpolate --load-config "$CFG" \
  --output-path "$PROJ/flythrough.mp4" \
  --pose-source train --interpolation-steps 8 --frame-rate 30 --output-format video

echo "=== DONE ==="
ls -la "$PROJ/export/splat.ply" "$PROJ/export/scene.ksplat" "$PROJ/flythrough.mp4"
