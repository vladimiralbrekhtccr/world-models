#!/bin/bash
set -e
cd /scratch/vladimir_albrekht/projects/world-models
source 3DGS/.venv/bin/activate
# Roll torch back to 2.4.0+cu124 and gsplat to the matching prebuilt wheel
PATH=$HOME/.local/bin:$PATH uv pip install --reinstall \
  "torch==2.4.0" "torchvision==0.19.0" \
  --index-url https://download.pytorch.org/whl/cu124
PATH=$HOME/.local/bin:$PATH uv pip install --reinstall --no-deps \
  gsplat==1.5.3 \
  --index-url https://docs.gsplat.studio/whl/pt24cu124/ \
  --extra-index-url https://pypi.org/simple/
python -c "import torch, gsplat; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('gsplat', gsplat.__version__)"
