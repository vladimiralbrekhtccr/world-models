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


def build_textured_mesh(depth: np.ndarray, panorama_path: Path, stride: int):
    """Build a textured triangle mesh from an equirectangular depth map.

    Each (downsampled) depth pixel becomes a vertex; quads of 4 adjacent
    vertices become 2 triangles. UVs are equirectangular, so the panorama
    image maps directly onto the mesh with no projection step needed.
    Returns a trimesh.Trimesh ready to export as .glb.
    """
    import trimesh
    d = depth[::stride, ::stride] if stride > 1 else depth
    H, W = d.shape

    # Vertex positions via spherical unprojection (same convention as PLY path).
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    lon = 2 * np.pi * (u + 0.5) / W - np.pi
    lat = np.pi / 2 - np.pi * (v + 0.5) / H
    cos_lat = np.cos(lat)
    x = d * cos_lat * np.sin(lon)
    y = d * np.sin(lat)
    z = d * cos_lat * np.cos(lon)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    # UVs: equirectangular → flat (u/W, 1 - v/H). 1 - v flips for image origin.
    uvs = np.stack(
        [(u + 0.5) / W, 1.0 - (v + 0.5) / H], axis=-1
    ).reshape(-1, 2).astype(np.float32)

    # Quad-grid triangulation. Inward-facing (wind triangles so the inside of
    # the spherical shell is the "front" — that's where the camera lives).
    rows = np.arange(H - 1)
    cols = np.arange(W - 1)
    r, c = np.meshgrid(rows, cols, indexing="ij")
    i00 = (r * W + c).reshape(-1)
    i01 = (r * W + c + 1).reshape(-1)
    i10 = ((r + 1) * W + c).reshape(-1)
    i11 = ((r + 1) * W + c + 1).reshape(-1)
    tris = np.empty((i00.size * 2, 3), dtype=np.int32)
    tris[0::2] = np.stack([i00, i11, i10], axis=1)   # inward-facing winding
    tris[1::2] = np.stack([i00, i01, i11], axis=1)

    img = Image.open(panorama_path).convert("RGB")
    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=img)
    return mesh


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panorama", default="panorama.png")
    p.add_argument("--out",      default="out/fast.ply")
    p.add_argument("--export",   choices=("ply", "glb", "both"), default="ply",
                   help="ply = 3DGS Gaussians; glb = textured triangle mesh; both = side-by-side outputs")
    p.add_argument("--stride",   type=int, default=1,
                   help="keep 1 of every N pixels (>1 = lighter PLY / coarser mesh)")
    p.add_argument("--near",     type=float, default=0.5)
    p.add_argument("--far",      type=float, default=30.0)
    p.add_argument("--scale_log",     type=float, default=-3.0,
                   help="log scale per Gaussian; -3.0 ≈ 5 cm radius")
    p.add_argument("--opacity_logit", type=float, default=2.0,
                   help="opacity in logit space; 2.0 ≈ 0.88 sigmoid")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Save depth viz next to the primary output
    viz = (255 * (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min() + 1e-8)).astype(np.uint8)
    Image.fromarray(viz).save(out_path.parent / "fast_depth.png")

    if args.export in ("ply", "both"):
        xyz, rgb = unproject_equirect(depth_m, color)
        xyz, rgb = downsample(xyz, rgb, stride=args.stride)
        ply_path = out_path if args.export == "ply" else out_path.with_suffix(".ply")
        print(f"[fast] {xyz.shape[0]:,} gaussians → {ply_path}")
        save_3dgs_ply(ply_path, xyz, rgb,
                      scale_log=args.scale_log, opacity_logit=args.opacity_logit)

    if args.export in ("glb", "both"):
        glb_path = out_path if args.export == "glb" else out_path.with_suffix(".glb")
        if glb_path.suffix.lower() != ".glb":
            glb_path = glb_path.with_suffix(".glb")
        mesh = build_textured_mesh(depth_m, Path(args.panorama), stride=args.stride)
        print(f"[fast] {len(mesh.vertices):,} verts · {len(mesh.faces):,} tris → {glb_path}")
        mesh.export(glb_path)


if __name__ == "__main__":
    main()
