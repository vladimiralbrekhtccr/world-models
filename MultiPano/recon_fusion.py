#!/usr/bin/env python3
"""Depth-fusion reconstruction from N equirectangular panoramas + poses.

Doesn't need gsplat / CUDA kernels — pure PyTorch + HuggingFace depth model.

For each panorama:
    pano                → DepthAnything-V2-Large (relative inverse depth)
    + pose (world XYZ)  → unproject each pixel via spherical coords
                          → world-space point cloud (xyz + rgb)

All clouds get concatenated and written as a canonical 3DGS PLY,
loadable by the existing /3dgs/ splat viewer or /3dgs-mesh/.

Coordinate convention (matches recon_3dgs.py):
  World: +X east, +Y up, +Z south. Pano center column = world-north (-Z).
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from transformers import pipeline

SH_C0 = 0.28209479177387814


def estimate_depth(img: Image.Image, device: int) -> np.ndarray:
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device,
    )
    out = pipe(img)
    return np.asarray(out["depth"], dtype=np.float32)


def to_metric(inv_depth: np.ndarray, near: float, far: float) -> np.ndarray:
    x = inv_depth.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    metric = 1.0 - x                        # 0 = closest
    return near + (far - near) * metric


def unproject_pano(depth: np.ndarray, color: np.ndarray, cam_pos: np.ndarray):
    """Return (N,3) world XYZ + (N,3) RGB in [0,1] for one panorama.

    World convention: +X east, +Y up, +Z south. Pano center column faces -Z."""
    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    lon = 2.0 * np.pi * (u + 0.5) / W - np.pi           # -π..π, 0 = center column = north
    lat = np.pi / 2.0 - np.pi * (v + 0.5) / H           # +π/2..-π/2
    cos_lat = np.cos(lat)
    dx =  cos_lat * np.sin(lon)                          # +X east
    dy =  np.sin(lat)                                    # +Y up
    dz = -cos_lat * np.cos(lon)                          # -Z at lon=0 (north)
    xyz_local = depth[..., None] * np.stack([dx, dy, dz], axis=-1)
    xyz_world = xyz_local + cam_pos
    rgb = color.astype(np.float32) / 255.0
    return xyz_world.reshape(-1, 3).astype(np.float32), rgb.reshape(-1, 3).astype(np.float32)


def downsample(xyz, rgb, stride):
    if stride <= 1: return xyz, rgb
    keep = np.arange(0, xyz.shape[0], stride)
    return xyz[keep], rgb[keep]


def save_3dgs_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray,
                  scale_log: float = -3.5, opacity_logit: float = 2.0):
    N = xyz.shape[0]
    f_dc = (rgb - 0.5) / SH_C0
    dtype = [
        ("x","f4"),("y","f4"),("z","f4"),
        ("nx","f4"),("ny","f4"),("nz","f4"),
        ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
        ("opacity","f4"),
        ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
        ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
    ]
    arr = np.zeros(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    arr["opacity"] = opacity_logit
    arr["scale_0"] = arr["scale_1"] = arr["scale_2"] = scale_log
    arr["rot_0"] = 1.0
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    print(f"[recon] wrote {path}  ({path.stat().st_size/1024/1024:.1f} MB · {N:,} gaussians)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--near", type=float, default=0.5)
    p.add_argument("--far",  type=float, default=18.0)
    p.add_argument("--camera-height", type=float, default=1.6)
    p.add_argument("--scale-log",     type=float, default=-3.5,
                   help="log of Gaussian scale; -3.5 ≈ 3 cm radius")
    args = p.parse_args()

    scene_dir = Path(args.scene)
    out = Path(args.out) if args.out else scene_dir / "scene.ply"

    poses = json.loads((scene_dir / "poses.json").read_text())
    scale = poses["scale_m_per_px"]
    cameras = poses["cameras"]

    # Map centre for world origin
    px_arr = [c["px"] for c in cameras.values()]
    py_arr = [c["py"] for c in cameras.values()]
    map_center_px = (max(px_arr) + min(px_arr)) / 2.0
    map_center_py = (max(py_arr) + min(py_arr)) / 2.0
    # Wider centre: use 2× max extents so origin is roughly map middle
    # (keep it simple — just centre on pin centroid)

    device = 0 if torch.cuda.is_available() else -1
    print(f"[recon] device = {'cuda:0' if device == 0 else 'cpu'}")

    all_xyz, all_rgb = [], []
    for pin_id in sorted(cameras.keys(), key=int):
        cam = cameras[pin_id]
        pano_path = scene_dir / f"pano_{pin_id}.png"
        if not pano_path.exists():
            sys.exit(f"missing {pano_path}")
        print(f"[recon] pin {pin_id}: loading {pano_path.name}")
        img = Image.open(pano_path).convert("RGB")
        color = np.asarray(img)
        print(f"  pano {color.shape[1]}×{color.shape[0]}")

        print("  running DepthAnything-V2-Large…")
        depth_raw = estimate_depth(img, device=device)
        depth_m = to_metric(depth_raw, near=args.near, far=args.far)

        # World position of camera
        x_w = (cam["px"] - map_center_px) * scale
        z_w = (cam["py"] - map_center_py) * scale
        y_w = args.camera_height
        cam_pos = np.array([x_w, y_w, z_w], dtype=np.float32)
        print(f"  camera at world ({x_w:.2f}, {y_w:.2f}, {z_w:.2f}) m")

        xyz, rgb = unproject_pano(depth_m, color, cam_pos)
        xyz, rgb = downsample(xyz, rgb, stride=args.stride)
        print(f"  → {xyz.shape[0]:,} points after stride {args.stride}")
        all_xyz.append(xyz); all_rgb.append(rgb)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    print(f"[recon] fused: {xyz.shape[0]:,} points across {len(cameras)} panoramas")
    save_3dgs_ply(out, xyz, rgb, scale_log=args.scale_log)


if __name__ == "__main__":
    main()
