#!/bin/bash
# Test WorldMirror 2.0 on a single example folder (multi-view → 3D recon).
# GPU 6 only, everything on scratch.
set -e

PROJ=/scratch/vladimir_albrekht/projects/hunyuanworld
PREFIX=$PROJ/.conda/hyworld2
REPO=$PROJ/HY-World-2.0
INPUT="${1:-$REPO/examples/worldrecon/realistic/Desk}"
OUTPUT="${2:-$PROJ/output/worldmirror/$(basename $INPUT)}"

# Caches → scratch
export HF_HOME=$PROJ/.cache/hf
export HF_TOKEN=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
export HF_HUB_TOKEN=$HF_TOKEN
export TORCH_HOME=$PROJ/.cache/torch
export XDG_CACHE_HOME=$PROJ/.cache/xdg
export TMPDIR=$PROJ/.tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $PREFIX

mkdir -p $OUTPUT
cd $REPO

echo "=== input: $INPUT ==="
echo "=== output: $OUTPUT ==="
echo "=== HF cache: $HF_HOME ==="
ls $INPUT | head -5
echo "..."

CUDA_VISIBLE_DEVICES=6 python -m hyworld2.worldrecon.pipeline \
  --input_path "$INPUT" \
  --output_path "$OUTPUT" \
  --enable_bf16

echo "=== WORLDMIRROR DONE ==="
find $OUTPUT -maxdepth 3 -type f | head -20
