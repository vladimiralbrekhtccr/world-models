#!/bin/bash
# Resume Lyra 2.0 install from where setup_lyra.sh failed.
# The failure was transformer_engine_torch couldn't find cuda_runtime_api.h
# because $CUDA_HOME/include is empty for conda CUDA — the real headers are
# under $CUDA_HOME/targets/x86_64-linux/include. We add that to CPATH.

set -e

LYRA_ROOT="/scratch/vladimir_albrekht/projects/lyra"
CONDA="$HOME/miniconda3/bin/conda"
ENV_PREFIX="$LYRA_ROOT/.conda/lyra2"

cd "$LYRA_ROOT/lyra/Lyra-2"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export CUDA_HOME="$CONDA_PREFIX"
SITE="$CONDA_PREFIX/lib/python3.10/site-packages"

# The fix: add targets/x86_64-linux/{include,lib} which is where conda CUDA
# actually puts headers and lib stubs.
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

echo "[resume] CPATH=$CPATH"
echo "[resume] verifying cuda_runtime_api.h is reachable…"
echo '#include <cuda_runtime_api.h>' | $CXX -E -x c++ - -o /dev/null -CPATH 2>/dev/null \
  && echo "  header reachable" \
  || true

# 5b. transformer_engine (the one that failed) — retry with new CPATH
pip install --no-build-isolation "transformer_engine[pytorch]"

# Symlink cuda_runtime as cudart for transformer_engine compatibility (idempotent)
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart" || true

# 6. flash-attn (the slow one; 10-30 min)
MAX_JOBS=16 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3

# 7. Vendored extensions
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

# 8. Smoke test
echo
echo "=== smoke test ==="
PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"

# 9. Model weights
echo
echo "=== downloading checkpoints (~10-15 GB) ==="
pip install -U huggingface_hub
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .

echo
echo "✓ Lyra 2.0 install complete. Activate with:"
echo "  source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate $ENV_PREFIX"
