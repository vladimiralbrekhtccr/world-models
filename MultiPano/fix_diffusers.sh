#!/bin/bash
cd /scratch/vladimir_albrekht/projects/world-models
source 3DGS/.venv/bin/activate
# Try diffusers 0.30.3 (released aug 2024, predates the torch.library changes in newer ones, but no cached_download dependency)
PATH=$HOME/.local/bin:$PATH uv pip install "diffusers==0.30.3" "huggingface_hub>=0.23,<0.26"
python -c "from diffusers import StableDiffusionImg2ImgPipeline; print('OK diffusers', __import__('diffusers').__version__)"
