#!/bin/bash
# Second attempt: also include the python-side nvidia/* includes which
# transformer_engine_torch reaches for (nvtx3/, cudnn/, etc.).
set -e

LYRA_ROOT="/scratch/vladimir_albrekht/projects/lyra"
ENV_PREFIX="$LYRA_ROOT/.conda/lyra2"

cd "$LYRA_ROOT/lyra/Lyra-2"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export CUDA_HOME="$CONDA_PREFIX"
SITE="$CONDA_PREFIX/lib/python3.10/site-packages"

# Cast the widest plausible header net.
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$SITE/nvidia/nvtx/include:$SITE/nvidia/cublas/include:$SITE/nvidia/cuda_runtime/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

echo "[r2] CPATH (newline-split):"
echo "$CPATH" | tr ':' '\n' | sed 's/^/  /'
echo "[r2] verifying nvtx3 reachable…"
echo '#include <nvtx3/nvToolsExt.h>' | $CXX -E -x c++ - -o /dev/null 2>&1 | head -3 && echo "  → found"

pip install --no-build-isolation "transformer_engine[pytorch]"
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart" || true

MAX_JOBS=16 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3

USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

echo "=== smoke ==="
PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"

pip install -U huggingface_hub
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
echo "✓ done"
