"""Run nerfstudio's official ExportGaussianSplat, bypassing the unrelated
pymeshlab import (which is broken in our env). Produces a PLY that the
mkkellogg viewer reads correctly — same convention as nerfstudio uses
internally."""
import sys, types, argparse
from pathlib import Path

# Stub pymeshlab BEFORE importing nerfstudio.scripts.exporter (which top-level
# imports it via nerfstudio.exporter.exporter_utils). The stub is never called
# for gaussian-splat export.
_stub = types.ModuleType("pymeshlab")
class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name): return _Any()
    def __call__(self, *a, **k): return _Any()
_stub.MeshSet = _Any
_stub.Mesh = _Any
sys.modules["pymeshlab"] = _stub

from nerfstudio.scripts.exporter import ExportGaussianSplat

ap = argparse.ArgumentParser()
ap.add_argument("--load-config", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--output-filename", default="splat.ply")
args = ap.parse_args()

exporter = ExportGaussianSplat(
    load_config=Path(args.load_config),
    output_dir=Path(args.output_dir),
    output_filename=args.output_filename,
    ply_color_mode="sh_coeffs",
)
exporter.main()
print(f"wrote {Path(args.output_dir) / args.output_filename}")
