#!/bin/bash
# Sweep WorldMirror 2.0 over the realistic examples, model loaded once.
# Skip 1-image folders (Flower, Office) — recon needs ≥2 views.
set -e

PROJ=/scratch/vladimir_albrekht/projects/hunyuanworld
PREFIX=$PROJ/.conda/hyworld2
REPO=$PROJ/HY-World-2.0
OUT=$PROJ/output/worldmirror_sweep

export HF_HOME=$PROJ/.cache/hf
export HF_TOKEN=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
export HF_HUB_TOKEN=$HF_TOKEN
export TORCH_HOME=$PROJ/.cache/torch
export XDG_CACHE_HOME=$PROJ/.cache/xdg
export TMPDIR=$PROJ/.tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $PREFIX
mkdir -p $OUT
cd $REPO

# Examples to sweep — skip Flower (1) and Office (1).
EXAMPLES=(Desk Dining_Table Park Workspace Room_Cat Valley Building Statue_Face Small_Room Messy_Room Park_Stone Tree_Building Landmark Archway_Tunnel)

FIRST="$REPO/examples/worldrecon/realistic/${EXAMPLES[0]}"
REST=("${EXAMPLES[@]:1}")
echo "=== sweep over ${#EXAMPLES[@]} examples → $OUT ==="

# Feed first as --input_path, rest via interactive stdin
{ for x in "${REST[@]}"; do echo "$REPO/examples/worldrecon/realistic/$x"; done; echo quit; } | \
CUDA_VISIBLE_DEVICES=6 python -m hyworld2.worldrecon.pipeline \
  --input_path "$FIRST" \
  --output_path "$OUT" \
  --enable_bf16

echo "=== SWEEP DONE ==="
echo "--- per-example PLY summary ---"
find $OUT -name gaussians.ply -printf "%s\t%p\n" | sort -nr
echo "--- per-example timing ---"
find $OUT -name pipeline_timing.json -exec sh -c 'echo "=== $1 ==="; cat $1' _ {} \;
