#!/bin/bash
set -e
cd /scratch/vladimir_albrekht/projects/world-models
source 3DGS/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python MultiPano/recon_fusion.py \
    --scene MultiPano/input/witch_hat_atelier \
    --stride 4 --near 0.5 --far 18
