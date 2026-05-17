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
    ap.add_argument("--opacity_bias", type=float, default=0.7,
                    help="Fraction of target drawn by opacity rank; rest is uniform random.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"loading {args.input}")
    ply = PlyData.read(str(args.input))
    v = ply["vertex"]
    n = len(v)
    print(f"  {n:,} gaussians, fields: {v.data.dtype.names}")
    target = min(args.target_count, n)
    if target == n:
        print("  already under target, copying as-is")
        ply.write(str(args.output))
        return

    rng = np.random.default_rng(args.seed)
    opacity = sigmoid(v["opacity"])
    n_bias = int(target * args.opacity_bias)
    n_rand = target - n_bias

    order = np.argsort(-opacity)
    bias_idx = order[:n_bias]
    remaining = np.setdiff1d(np.arange(n), bias_idx, assume_unique=False)
    rand_idx = rng.choice(remaining, size=n_rand, replace=False)
    keep = np.concatenate([bias_idx, rand_idx])
    keep.sort()
    print(f"  keep {len(keep):,} / {n:,}  (bias={n_bias:,} + random={n_rand:,})")

    new = v.data[keep]
    el = PlyElement.describe(new, "vertex")
    PlyData([el], text=False).write(str(args.output))
    size = args.output.stat().st_size / 1e6
    print(f"  wrote {args.output}  ({size:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
