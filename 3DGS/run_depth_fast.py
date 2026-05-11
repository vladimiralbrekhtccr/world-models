#!/usr/bin/env python3
"""
Path A — fast panorama-to-3DGS using monocular depth on the equirectangular
panorama, with no diffusion refinement.

Pipeline:
    panorama.png
        -> DepthAnything-V2-Large  (relative inverse depth)
        -> spherical unprojection  (one Gaussian per pixel)
        -> out/fast.ply            (canonical 3DGS PLY)

The resulting .ply opens in any 3DGS viewer (SuperSplat, antimatter15/splat,
gsplat.js, etc.). Quality is good near the original camera, poor when the
viewer moves far from it because there is no information about what's behind
foreground objects.

Run:
    python run_depth_fast.py
    python run_depth_fast.py --panorama other.png --out out/other.ply

Output:
    out/fast.ply
    out/fast_depth.png    (visualization of the predicted depth)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from plyfile import PlyData, PlyElement
from transformers import pipeline

SH_C0 = 0.28209479177387814  # zero-band SH normalization, used by 3DGS PLYs


def estimate_depth(img: Image.Image, device: int) -> np.ndarray:
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device,
    )
    out = pipe(img)
    depth_img = out["depth"]  # PIL image, mode 'I;16' or float
    depth = np.asarray(depth_img, dtype=np.float32)
    return depth


def to_metric_range(inv_depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Treat DepthAnything output as relative inverse depth (large = near).
    Invert, normalize, rescale to a chosen [near, far] window in metres.
    Absolute scale here is meaningless — it just sets a sensible Gaussian density."""
    x = inv_depth.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)   # [0, 1], 1 = closest
    metric = (1.0 - x)                                # [0, 1], 0 = closest
    return near + (far - near) * metric


def unproject_equirect(depth: np.ndarray, color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map equirectangular pixels to 3D using spherical coords. Returns (N,3), (N,3)."""
    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    lon = 2 * np.pi * (u + 0.5) / W - np.pi             # -pi .. pi
    lat = np.pi / 2 - np.pi * (v + 0.5) / H             # +pi/2 .. -pi/2
    cos_lat = np.cos(lat)
    x = depth * cos_lat * np.sin(lon)
    y = depth * np.sin(lat)
    z = depth * cos_lat * np.cos(lon)
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    rgb = color.reshape(-1, 3).astype(np.float32) / 255.0
    return xyz, rgb


def downsample(xyz: np.ndarray, rgb: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if stride <= 1:
        return xyz, rgb
    keep = np.arange(0, xyz.shape[0], stride)
    return xyz[keep], rgb[keep]


def save_3dgs_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray,
                  scale_log: float, opacity_logit: float) -> None:
    """Write a minimal but canonical 3DGS PLY (xyz, normals, DC SH, opacity,
    log-scale, identity rotation). Most 3DGS viewers will load it."""
    N = xyz.shape[0]
    f_dc = (rgb - 0.5) / SH_C0

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    arr = np.zeros(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    arr["opacity"] = opacity_logit
    arr["scale_0"] = arr["scale_1"] = arr["scale_2"] = scale_log
    arr["rot_0"] = 1.0  # identity quaternion (w=1, xyz=0)

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(str(path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panorama", default="panorama.png")
    p.add_argument("--out",      default="out/fast.ply")
    p.add_argument("--stride",   type=int, default=1,
                   help="keep 1 of every N pixels (>1 = lighter PLY)")
    p.add_argument("--near",     type=float, default=0.5)
    p.add_argument("--far",      type=float, default=30.0)
    p.add_argument("--scale_log",     type=float, default=-3.0,
                   help="log scale per Gaussian; -3.0 ≈ 5 cm radius")
    p.add_argument("--opacity_logit", type=float, default=2.0,
                   help="opacity in logit space; 2.0 ≈ 0.88 sigmoid")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1
    print(f"[fast] device={'cuda:0' if device == 0 else 'cpu'}")
    print(f"[fast] loading {args.panorama}")
    img = Image.open(args.panorama).convert("RGB")
    color = np.asarray(img)
    print(f"[fast] image {color.shape[1]}x{color.shape[0]}")

    print("[fast] running DepthAnything-V2-Large …")
    depth_raw = estimate_depth(img, device=device)
    print(f"[fast] raw depth range [{depth_raw.min():.3f}, {depth_raw.max():.3f}]")

    depth_m = to_metric_range(depth_raw, near=args.near, far=args.far)
    print(f"[fast] mapped to [{args.near}, {args.far}] m")

    # Save depth viz
    viz = (255 * (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min() + 1e-8)).astype(np.uint8)
    Image.fromarray(viz).save(Path(args.out).parent / "fast_depth.png")

    xyz, rgb = unproject_equirect(depth_m, color)
    xyz, rgb = downsample(xyz, rgb, stride=args.stride)
    print(f"[fast] {xyz.shape[0]:,} gaussians")

    save_3dgs_ply(Path(args.out), xyz, rgb,
                  scale_log=args.scale_log, opacity_logit=args.opacity_logit)
    print(f"[fast] wrote {args.out}")


if __name__ == "__main__":
    main()
