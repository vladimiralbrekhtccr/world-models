#!/usr/bin/env bash
# Path B — DreamScene360 (Zhou et al., ECCV 2024).
# Clones the upstream repo into ./third_party, installs extra deps into the
# active conda env, runs training with our panorama.
#
# Verify DS360_REPO before running — repo may have moved.
# Expected wall time: ~1-3 h on a single H100 80GB.

set -euo pipefail

DS360_REPO="${DS360_REPO:-https://github.com/ShijieZhou-UCLA/DreamScene360.git}"
DS360_DIR="${DS360_DIR:-third_party/DreamScene360}"
PANORAMA="${PANORAMA:-$(pwd)/panorama.png}"
OUT_DIR="${OUT_DIR:-$(pwd)/out/dreamscene360}"

mkdir -p "$(dirname "$DS360_DIR")" "$OUT_DIR"

if [[ ! -d "$DS360_DIR" ]]; then
  echo "[dreamscene360] cloning $DS360_REPO -> $DS360_DIR"
  git clone --depth=1 "$DS360_REPO" "$DS360_DIR"
fi

# Extra deps that DreamScene360 typically needs on top of env_setup.sh's stack.
# Install only if not already present.
pip install --quiet --no-input \
  einops kornia trimesh open3d \
  ninja scipy matplotlib \
  || true

cd "$DS360_DIR"

# Build any CUDA extensions the repo ships with. Most 3DGS forks have a
# `submodules/diff-gaussian-rasterization` and a `simple-knn` — these need
# to be pip-installed against the repo's CUDA build.
if [[ -d submodules ]]; then
  for ext in submodules/*/; do
    if [[ -f "$ext/setup.py" ]]; then
      echo "[dreamscene360] building extension: $ext"
      pip install -e "$ext"
    fi
  done
fi

# The exact entry point varies by upstream commit. Try the documented ones in
# order; first one that exists wins.
ENTRYPOINTS=(
  "scripts/run_panorama.py"
  "train.py"
  "main.py"
)
ENTRY=""
for e in "${ENTRYPOINTS[@]}"; do
  if [[ -f "$e" ]]; then ENTRY="$e"; break; fi
done

if [[ -z "$ENTRY" ]]; then
  echo "[dreamscene360] ERROR: could not find a known entry point in $DS360_DIR"
  echo "[dreamscene360] inspect the repo and edit this script — common scripts:"
  ls -la
  exit 1
fi

echo "[dreamscene360] running: python $ENTRY  --panorama=$PANORAMA  --out=$OUT_DIR"
python "$ENTRY" \
  --panorama "$PANORAMA" \
  --output_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/train.log"

echo "[dreamscene360] done. .ply expected under $OUT_DIR"
