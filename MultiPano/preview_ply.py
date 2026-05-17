#!/usr/bin/env python3
"""Render a few preview frames from a 3DGS PLY using gsplat.

Renders 4 perspective views (N, E, S, W) from a chosen camera position,
saves them as PNG. Quick smoke test for what the scene actually looks
like, without needing the browser viewer.
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData
import gsplat

SH_C0 = 0.28209479177387814

def load_ply(path):
    ply = PlyData.read(str(path)); v = ply["vertex"]
    pos = torch.tensor(np.stack([v["x"], v["y"], v["z"]], -1), dtype=torch.float32)
    quats = torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], -1), dtype=torch.float32)
    scales = torch.tensor(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1), dtype=torch.float32)
    opac = torch.tensor(v["opacity"], dtype=torch.float32)
    fdc = torch.tensor(np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1), dtype=torch.float32)
    rgb = (fdc * SH_C0 + 0.5).clamp(1e-3, 1 - 1e-3)
    return pos, quats, scales, opac, rgb

def make_viewmat(eye, yaw, pitch=0.0):
    fx, fy, fz = math.sin(yaw)*math.cos(pitch), math.sin(pitch), -math.cos(yaw)*math.cos(pitch)
    forward = np.array([fx, fy, fz], dtype=np.float64)
    right = np.cross(forward, [0, 1, 0]); right /= np.linalg.norm(right)
    cam_down = np.cross(right, forward)
    R = np.stack([right, cam_down, forward], axis=0).astype(np.float32)
    t = -R @ np.asarray(eye, dtype=np.float64)
    M = np.eye(4, dtype=np.float32); M[:3,:3] = R; M[:3,3] = t.astype(np.float32)
    return M

def make_K(size, fov):
    f = (size/2) / math.tan(math.radians(fov)/2)
    return np.array([[f,0,size/2],[0,f,size/2],[0,0,1]], dtype=np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--fov", type=float, default=90)
    p.add_argument("--eye", default=None, help="x,y,z eye position; defaults to PLY centroid")
    p.add_argument("--yaws", default="0,90,180,270", help="comma list of yaw angles in deg")
    args = p.parse_args()

    ply = Path(args.ply)
    out_dir = Path(args.out_dir) if args.out_dir else ply.parent / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")

    pos, quats, scales, opac, rgb = (t.to(dev) for t in load_ply(ply))
    quats_n = F.normalize(quats, dim=-1)
    scales_p = torch.exp(scales); opac_p = torch.sigmoid(opac)
    print(f"loaded {len(pos):,} gaussians")

    if args.eye:
        eye = np.array([float(x) for x in args.eye.split(",")])
    else:
        eye = pos.cpu().numpy().mean(axis=0); eye[1] = 1.6   # eye height
    print(f"camera eye = {eye}")

    K = torch.tensor(make_K(args.size, args.fov), device=dev)[None]
    yaws = [math.radians(float(y)) for y in args.yaws.split(",")]
    for i, yaw in enumerate(yaws):
        viewmat = torch.tensor(make_viewmat(eye, yaw), device=dev)[None]
        img, _, _ = gsplat.rasterization(pos, quats_n, scales_p, opac_p, rgb,
                                          viewmat, K, args.size, args.size,
                                          packed=False, render_mode="RGB")
        arr = (img[0].clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
        out_path = out_dir / f"{ply.stem}_yaw{int(math.degrees(yaw)):03d}.png"
        Image.fromarray(arr).save(out_path)
        print(f"  → {out_path.name}")

if __name__ == "__main__":
    main()
