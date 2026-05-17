"""Render a PLY from a specified (eye, look_at, up) pose using gsplat.

Used to debug the browser viewer: render with the same pose the viewer
starts at, so we can see what the user sees without a browser.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
import gsplat

SH_C0 = 0.28209479177387814


def look_at_matrix(eye, target, up):
    """Right-handed view matrix where camera +Z faces *target* (OpenCV / Lyra)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_down = np.cross(forward, right)  # because up is world-down (y-down frame)
    R = np.stack([right, cam_down, forward], axis=0).astype(np.float32)
    t = -R @ eye
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t.astype(np.float32)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("out")
    ap.add_argument("--eye", default="0,-2,-3")
    ap.add_argument("--look_at", default="0,0,10")
    ap.add_argument("--up", default="0,-1,0")
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--H", type=int, default=540)
    ap.add_argument("--hfov", type=float, default=70.0)
    args = ap.parse_args()

    print(f"loading {args.ply}")
    v = PlyData.read(args.ply)["vertex"]
    n = len(v)
    print(f"  {n:,} gaussians")
    pos = torch.tensor(np.stack([v["x"], v["y"], v["z"]], -1), dtype=torch.float32).cuda()
    quats = torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], -1), dtype=torch.float32).cuda()
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.tensor(np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1)), dtype=torch.float32).cuda()
    opac = torch.tensor(1.0 / (1.0 + np.exp(-v["opacity"])), dtype=torch.float32).cuda()
    fdc = torch.tensor(np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1), dtype=torch.float32).cuda()
    rgb = (fdc * SH_C0 + 0.5).clamp(0, 1)

    eye = [float(x) for x in args.eye.split(",")]
    target = [float(x) for x in args.look_at.split(",")]
    up = [float(x) for x in args.up.split(",")]
    viewmat = torch.from_numpy(look_at_matrix(eye, target, up)).cuda().unsqueeze(0)
    fx = (args.W / 2) / math.tan(math.radians(args.hfov) / 2)
    K = torch.tensor([[fx, 0, args.W / 2], [0, fx, args.H / 2], [0, 0, 1]], dtype=torch.float32).cuda().unsqueeze(0)

    img, _, _ = gsplat.rasterization(
        means=pos, quats=quats, scales=scales, opacities=opac, colors=rgb,
        viewmats=viewmat, Ks=K, width=args.W, height=args.H,
        sh_degree=None, render_mode="RGB",
    )
    arr = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(args.out)
    print(f"wrote {args.out}  eye={eye}  look_at={target}  up={up}  hfov={args.hfov}")


if __name__ == "__main__":
    main()
