"""Downsample a 3DGS PLY so it fits GitHub Pages (~80 MB).

Strategy: opacity-weighted subsample. Keep `--target_count` Gaussians, biasing
selection toward higher-sigmoid(opacity) splats (visually most important),
plus a small uniform random fraction so we don't lose spatial coverage.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--target_count", type=int, default=1_200_000)
    ap.add_argument("--min_opacity", type=float, default=0.1,
                    help="Drop gaussians with sigmoid(opacity) below this.")
    ap.add_argument("--max_radius", type=float, default=100.0,
                    help="Drop gaussians whose ||xyz|| exceeds this (floater pruning).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"loading {args.input}")
    ply = PlyData.read(str(args.input))
    v = ply["vertex"]
    n = len(v)
    print(f"  {n:,} gaussians, fields: {v.data.dtype.names}")

    opacity = sigmoid(v["opacity"])
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    radius = np.linalg.norm(xyz, axis=1)
    pool = np.where((opacity > args.min_opacity) & (radius < args.max_radius))[0]
    print(f"  pool after opacity>{args.min_opacity} + radius<{args.max_radius}: "
          f"{len(pool):,} ({len(pool)/n*100:.1f}%)")

    target = min(args.target_count, len(pool))
    rng = np.random.default_rng(args.seed)
    keep = rng.choice(pool, size=target, replace=False)
    keep.sort()
    print(f"  uniform-random keep {len(keep):,}")

    new = v.data[keep]
    el = PlyElement.describe(new, "vertex")
    PlyData([el], text=False).write(str(args.output))
    size = args.output.stat().st_size / 1e6
    print(f"  wrote {args.output}  ({size:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
