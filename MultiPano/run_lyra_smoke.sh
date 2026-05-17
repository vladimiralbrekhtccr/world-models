#!/bin/bash
set -e
ENV_PREFIX=/scratch/vladimir_albrekht/projects/lyra/.conda/lyra2
LYRA=/scratch/vladimir_albrekht/projects/lyra/lyra/Lyra-2
cd $LYRA
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_PREFIX
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cufft/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use DMD distillation LoRA — 4-step sampling, ~35s instead of 9min.
# Sample 4 from their assets/samples/.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference \
  --input_image_path assets/samples \
  --sample_id 4 \
  --experiment lyra2 \
  --checkpoint_dir checkpoints/model \
  --prompt_dir assets/samples \
  --output_path outputs/zoomgs \
  --num_frames_zoom_in 81 \
  --num_frames_zoom_out 81 \
  --zoom_in_strength 0.5 \
  --zoom_out_strength 1.5 \
  --use_dmd
echo "=== step 1 (video gen) done. Now step 2 (3DGS recon) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon \
  --input_video_path outputs/zoomgs/videos/4.mp4

echo "=== ALL DONE ==="
ls -la outputs/zoomgs/4_gs_ours/ 2>&1 || echo "no _gs_ours dir"
find outputs -name "*.ply" 2>&1
