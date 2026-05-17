#!/bin/bash
cd /scratch/vladimir_albrekht/projects/world-models
source 3DGS/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python MultiPano/recon_3dgs.py \
    --scene MultiPano/input/witch_hat_atelier \
    --iters 6000 --num-yaw 8 --num-pitch 3 --view-size 384 --n-init 20000
