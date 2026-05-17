#!/bin/bash
cd /scratch/vladimir_albrekht/projects/world-models
source 3DGS/.venv/bin/activate
echo === gsplat package files ===
ls 3DGS/.venv/lib/python3.11/site-packages/gsplat/cuda/ | head -10
echo === look for .so ===
find 3DGS/.venv/lib/python3.11/site-packages/gsplat -name "*.so" 2>/dev/null
echo === try direct wheel download ===
PATH=$HOME/.local/bin:$PATH pip download --no-deps --dest /tmp/gswheel gsplat==1.5.3 -i https://docs.gsplat.studio/whl/pt24cu124/ 2>&1 | tail -8
ls /tmp/gswheel/ 2>&1
