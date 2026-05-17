#!/bin/bash
cd /scratch/vladimir_albrekht/projects/world-models
source 3DGS/.venv/bin/activate
PATH=$HOME/.local/bin:$PATH uv pip install --reinstall --index-strategy=unsafe-best-match \
  gsplat \
  -i https://docs.gsplat.studio/whl/pt24cu124/ \
  --extra-index-url https://pypi.org/simple/
