"""Merge multiple Lyra-2 PLY outputs into a single world-frame PLY.

Each Lyra run defines its own coordinate frame (camera-0 at origin looking +Z
in OpenCV / Y-down). To merge runs that share the same camera-0 *position*
but differ in *yaw*, rotate each PLY by R_y(yaw) so its local +Z aligns
with the world direction it was aimed at.

Inputs: per-run (ply_path, yaw_deg). Each run filtered (low-opacity + far)
then concatenated. The merged PLY is uniformly subsampled to fit GH Pages.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rotate_y(verts: np.ndarray, theta_rad: float) -> np.ndarray:
    """In-place rotate xyz, nx/ny/nz and rot quaternion by R_y(theta)."""
    c, s = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    # positions and normals: R_y rotates (x,z), leaves y alone
    for px, pz in (("x", "z"), ("nx", "nz")):
        x = verts[px].copy()
        z = verts[pz].copy()
        verts[px] = c * x + s * z
        verts[pz] = -s * x + c * z
    # quaternion compose: q_new = q_axis * q_old   with q_axis = (W,0,Y,0)
    W = float(np.cos(theta_rad / 2.0))
    Y = float(np.sin(theta_rad / 2.0))
    qw, qx, qy, qz = (verts[f"rot_{k}"].copy() for k in range(4))
    verts["rot_0"] = W * qw - Y * qy
    verts["rot_1"] = W * qx + Y * qz
    verts["rot_2"] = W * qy + Y * qw
    verts["rot_3"] = W * qz - Y * qx
    return verts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="ply_path:yaw_deg  (repeat per run)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target_count", type=int, default=1_500_000)
    ap.add_argument("--min_opacity", type=float, default=0.1)
    ap.add_argument("--max_radius", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    chunks = []
    for spec in args.input:
        ply_path, yaw_str = spec.split(":")
        yaw = float(yaw_str)
        print(f"\nloading {ply_path}  (yaw {yaw}°)")
        v = PlyData.read(ply_path)["vertex"].data.copy()
        print(f"  {len(v):,} gaussians")
        op = sigmoid(v["opacity"])
        r = np.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
        pool = np.where((op > args.min_opacity) & (r < args.max_radius))[0]
        print(f"  after opacity>{args.min_opacity} + r<{args.max_radius}: {len(pool):,}")
        v = v[pool]
        if yaw != 0.0:
            v = rotate_y(v, float(np.deg2rad(yaw)))
            print(f"  rotated by {yaw}° around Y")
        chunks.append(v)

    merged = np.concatenate(chunks)
    print(f"\nmerged total: {len(merged):,}")
    target = min(args.target_count, len(merged))
    if target < len(merged):
        idx = rng.choice(len(merged), size=target, replace=False)
        idx.sort()
        merged = merged[idx]
        print(f"uniform-subsample to {target:,}")

    el = PlyElement.describe(merged, "vertex")
    PlyData([el], text=False).write(str(args.output))
    print(f"\nwrote {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
