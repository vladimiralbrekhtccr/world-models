#!/bin/bash
# COLMAP dense MVS pipeline on the already-processed nerfstudio sparse recon.
#  1. image_undistorter   — rectify images to PINHOLE
#  2. patch_match_stereo  — per-image depth maps (GPU, slow: ~30-60 min)
#  3. stereo_fusion       — fused colored point cloud
#  4. poisson_mesher      — triangle mesh
set -e

PROC=/scratch/vladimir_albrekht/projects/world-models/MultiPano/ns_room2/processed
DENSE=$PROC/dense
PREFIX=/scratch/vladimir_albrekht/projects/world-models/MultiPano/.conda/nerfstudio
LYRA=/scratch/vladimir_albrekht/projects/lyra/.conda/lyra2

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $PREFIX
# colmap needs libcudart.so.12 which lives in lyra (conda-forge cuda runtime)
export LD_LIBRARY_PATH=$PREFIX/lib:$LYRA/lib:$LD_LIBRARY_PATH

mkdir -p $DENSE
cd $PROC

echo "=== 1/4: image_undistorter ==="
CUDA_VISIBLE_DEVICES=0 colmap image_undistorter \
  --image_path $PROC/images \
  --input_path $PROC/colmap/sparse/0 \
  --output_path $DENSE \
  --output_type COLMAP

echo "=== 2/4: patch_match_stereo (GPU, slow) ==="
CUDA_VISIBLE_DEVICES=0 colmap patch_match_stereo \
  --workspace_path $DENSE \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.num_iterations 5

echo "=== 3/4: stereo_fusion ==="
CUDA_VISIBLE_DEVICES=0 colmap stereo_fusion \
  --workspace_path $DENSE \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path $DENSE/fused.ply

echo "=== 4/4: poisson_mesher ==="
colmap poisson_mesher \
  --input_path $DENSE/fused.ply \
  --output_path $DENSE/meshed-poisson.ply \
  --PoissonMeshing.depth 11 \
  --PoissonMeshing.trim 7

echo "=== DONE ==="
ls -la $DENSE/fused.ply $DENSE/meshed-poisson.ply 2>&1
