#!/usr/bin/env python3
"""Multi-view 3D Gaussian Splatting from N equirectangular panoramas + poses.

Pipeline:
    panoramas + poses.json
        ─►  slice each pano into a fan of perspective views (known intrinsics + extrinsics)
        ─►  train 3DGS with photometric loss (gsplat 1.x + DefaultStrategy)
        ─►  out/scene.ply (canonical 3DGS PLY, loadable by mkkellogg viewer)

Run on a node with a single H100 visible:
    ssh node001
    cd /scratch/vladimir_albrekht/projects/world-models
    source 3DGS/.venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python MultiPano/recon_3dgs.py \\
        --scene MultiPano/input/witch_hat_atelier

Coordinate conventions
----------------------
World:   +X east, +Y up, +Z south (so map-north = -Z).
Camera:  OpenCV (+X right, +Y down, +Z forward), per gsplat.
Panorama: equirectangular; centre column faces world -Z (north),
          left→west, right→east, seam at the back = south.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

import gsplat
from gsplat import DefaultStrategy

SH_C0 = 0.28209479177387814   # zero-band SH normalization for the canonical 3DGS PLY


# ── world / camera math ───────────────────────────────────────────
def look_at_R(yaw: float, pitch: float) -> np.ndarray:
    """Return R_world_to_cam (3x3) for a camera at the origin looking in
    direction (yaw, pitch). yaw=0 → looking world -Z (north)."""
    fx = math.sin(yaw) * math.cos(pitch)
    fy = math.sin(pitch)
    fz = -math.cos(yaw) * math.cos(pitch)
    forward = np.array([fx, fy, fz], dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    n = np.linalg.norm(right)
    if n < 1e-9:                      # looking straight up/down — pick an arbitrary right
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / n
    cam_down = np.cross(right, forward)
    R = np.stack([right, cam_down, forward], axis=0).astype(np.float32)   # rows = cam axes in world
    return R


def make_viewmat(position: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    R = look_at_R(yaw, pitch)
    t = -R @ position.astype(np.float64)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3]  = t.astype(np.float32)
    return M


def make_K(size: int, fov_deg: float) -> np.ndarray:
    f = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[f, 0, size / 2.0],
                  [0, f, size / 2.0],
                  [0, 0, 1.0]], dtype=np.float32)
    return K


def sample_pano(pano: np.ndarray, R_world_to_cam: np.ndarray,
                fov_deg: float, size: int) -> np.ndarray:
    """Sample an equirectangular panorama into a perspective view.
    Returns float32 HxWx3 in [0, 1]."""
    H, W = pano.shape[:2]
    u, v = np.meshgrid(np.arange(size), np.arange(size))
    f = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx = cy = size / 2.0
    x = (u - cx) / f
    y = (v - cy) / f
    z = np.ones_like(x)
    d_cam = np.stack([x, y, z], axis=-1).astype(np.float64)
    d_cam = d_cam / np.linalg.norm(d_cam, axis=-1, keepdims=True)
    # world = R^T @ camera
    d_world = d_cam @ R_world_to_cam.astype(np.float64)
    dx, dy, dz = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lon = np.arctan2(dx, -dz)                  # 0 = north
    lat = np.arcsin(np.clip(dy, -1.0, 1.0))    # +π/2 = up
    sx = (lon + math.pi) / (2 * math.pi) * W
    sy = (math.pi / 2.0 - lat) / math.pi * H

    # bilinear sample with horizontal wrap (longitude wraps)
    sx_w = np.mod(sx, W)                       # wrap
    sx0 = np.floor(sx_w).astype(int) % W
    sx1 = (sx0 + 1) % W
    sy0 = np.clip(np.floor(sy).astype(int), 0, H - 1)
    sy1 = np.clip(sy0 + 1, 0, H - 1)
    fx = (sx_w - np.floor(sx_w))[..., None]
    fy = (sy - np.floor(sy))[..., None]
    p00 = pano[sy0, sx0].astype(np.float32)
    p10 = pano[sy0, sx1].astype(np.float32)
    p01 = pano[sy1, sx0].astype(np.float32)
    p11 = pano[sy1, sx1].astype(np.float32)
    out = p00 * (1 - fx) * (1 - fy) + p10 * fx * (1 - fy) + \
          p01 * (1 - fx) * fy       + p11 * fx * fy
    return (out / 255.0).astype(np.float32)


# ── scene loader / view fan ───────────────────────────────────────
def build_views(scene_dir: Path, num_yaw: int, num_pitch: int,
                fov_deg: float, view_size: int, camera_height: float):
    poses_path = scene_dir / "poses.json"
    if not poses_path.exists():
        sys.exit(f"poses.json not found at {poses_path}")
    poses = json.loads(poses_path.read_text())
    scale = poses["scale_m_per_px"]
    cameras = poses["cameras"]

    # World origin = map centre (so coords are centred)
    # We need the map size — infer from one of the panos? Use the largest pin coord roughly.
    px_arr = [c["px"] for c in cameras.values()]
    py_arr = [c["py"] for c in cameras.values()]
    map_extent = max(max(px_arr), max(py_arr)) * 2  # very rough; just for centring
    map_center_px = map_extent / 2.0

    views = []                  # list of dicts: image (Tensor HxWx3 [0,1]), viewmat (4x4), pos (3,)
    pin_world_positions = {}
    for pin_id in sorted(cameras.keys(), key=int):
        cam = cameras[pin_id]
        pano_path = scene_dir / f"pano_{pin_id}.png"
        if not pano_path.exists():
            sys.exit(f"missing {pano_path}")
        pano = np.asarray(Image.open(pano_path).convert("RGB"))
        # World position of the camera (centre map at origin)
        x_world = (cam["px"] - map_center_px) * scale
        z_world = (cam["py"] - map_center_px) * scale
        y_world = camera_height
        pos = np.array([x_world, y_world, z_world], dtype=np.float32)
        pin_world_positions[pin_id] = pos

        # Generate the fan of perspective views
        yaw_base = math.radians(cam.get("yaw_deg", 0))
        for i in range(num_yaw):
            yaw = yaw_base + i * 2 * math.pi / num_yaw
            for j in range(num_pitch):
                if num_pitch == 1:
                    pitch = 0.0
                else:
                    # spread pitches from -25° to +25°
                    pitch = math.radians(-25 + 50 * j / (num_pitch - 1))
                R = look_at_R(yaw, pitch)
                img = sample_pano(pano, R, fov_deg, view_size)
                vm = np.eye(4, dtype=np.float32)
                vm[:3, :3] = R
                vm[:3, 3]  = (-R @ pos.astype(np.float64)).astype(np.float32)
                views.append({"image": img, "viewmat": vm, "pos": pos, "pin": pin_id})
    print(f"[recon] built {len(views)} perspective views from {len(cameras)} panoramas")
    return views, pin_world_positions, scale


# ── 3DGS init + train ─────────────────────────────────────────────
def init_splats(scene_center: np.ndarray, scene_radius: float, n_init: int, device,
                init_ply: Path = None):
    """Initialise splats either from a sparse-init PLY (positions + colors from
    depth-fusion) or from random points in a ball. Random init produces
    over-smoothed garbage for sparse-view setups; sparse-init is the standard
    3DGS practice and dramatically improves convergence."""
    if init_ply and Path(init_ply).exists():
        print(f"[init] sparse-init from {init_ply}")
        ply = PlyData.read(str(init_ply))
        v = ply["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        # Recover RGB from f_dc (canonical 3DGS PLY uses f_dc = (rgb - 0.5) / SH_C0)
        fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
        rgb = np.clip(fdc * SH_C0 + 0.5, 1e-3, 1.0 - 1e-3)
        # Downsample if too many points
        if len(pts) > n_init:
            keep = np.random.default_rng(0).choice(len(pts), n_init, replace=False)
            pts = pts[keep]; rgb = rgb[keep]
        N = len(pts)
        means_init = torch.tensor(pts, dtype=torch.float32, device=device)
        # Logit of the color so sigmoid(stored) = rgb
        logit_rgb = np.log(rgb / (1.0 - rgb))
        colors_init = torch.tensor(logit_rgb, dtype=torch.float32, device=device)
        scales_init = torch.full((N, 3), math.log(0.05), device=device)   # ~5 cm
        quats_init = torch.zeros(N, 4, device=device); quats_init[:, 0] = 1.0
        opacities_init = torch.full((N,), float(np.log(0.5 / 0.5)), device=device)  # logit(0.5)
    else:
        print(f"[init] random ball around scene centre · {n_init} points")
        rng = np.random.default_rng(0)
        direc = rng.standard_normal((n_init, 3))
        direc /= np.linalg.norm(direc, axis=1, keepdims=True) + 1e-9
        r = scene_radius * np.cbrt(rng.uniform(0.05, 1.0, n_init))
        pts = scene_center + direc * r[:, None]
        N = n_init
        means_init = torch.tensor(pts, dtype=torch.float32, device=device)
        scales_init = torch.full((N, 3), math.log(0.1), device=device)
        quats_init = torch.zeros(N, 4, device=device); quats_init[:, 0] = 1.0
        opacities_init = torch.full((N,), float(np.log(0.1 / 0.9)), device=device)
        colors_init = torch.full((N, 3), 0.0, device=device)

    splats = nn.ParameterDict({
        "means":     nn.Parameter(means_init),
        "scales":    nn.Parameter(scales_init),
        "quats":     nn.Parameter(quats_init),
        "opacities": nn.Parameter(opacities_init),
        "colors":    nn.Parameter(colors_init),
    })
    return splats


def ssim_1d(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Simple SSIM over an HxWx3 image (no batch dim). Returns scalar in [0,1]."""
    # Move channels to front and add batch
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    y = y.permute(2, 0, 1).unsqueeze(0)
    pad = window_size // 2
    sigma = 1.5
    coords = torch.arange(window_size, device=x.device, dtype=torch.float32) - pad
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).view(1, 1, 1, -1)
    g = g.expand(3, 1, 1, window_size)
    gT = g.transpose(2, 3)
    def conv(t):
        t = F.conv2d(t, g, padding=(0, pad), groups=3)
        t = F.conv2d(t, gT, padding=(pad, 0), groups=3)
        return t
    mu_x, mu_y = conv(x), conv(y)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sig_x2 = conv(x*x) - mu_x2
    sig_y2 = conv(y*y) - mu_y2
    sig_xy = conv(x*y) - mu_xy
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_xy + C1) * (2*sig_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    return ssim_map.mean()


def make_optimizers(splats: nn.ParameterDict, scene_radius: float):
    return {
        "means":     torch.optim.Adam([splats["means"]],     lr=1.6e-4 * scene_radius),
        "scales":    torch.optim.Adam([splats["scales"]],    lr=5e-3),
        "quats":     torch.optim.Adam([splats["quats"]],     lr=1e-3),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=5e-2),
        "colors":    torch.optim.Adam([splats["colors"]],    lr=2.5e-3),
    }


def render_view(splats, viewmat, K, W, H):
    quats_n = F.normalize(splats["quats"], dim=-1)
    scales_p = torch.exp(splats["scales"])
    opacities_p = torch.sigmoid(splats["opacities"])
    colors_p = torch.sigmoid(splats["colors"])
    img, alpha, info = gsplat.rasterization(
        splats["means"], quats_n, scales_p, opacities_p, colors_p,
        viewmat[None], K[None], W, H,
        packed=False, render_mode="RGB",
    )
    return img.squeeze(0), info


def train(views, scene_center, scene_radius, n_init, n_iters, fov_deg, view_size,
          out_ply, device, log_every=200, init_ply: Path = None, ssim_weight: float = 0.2):
    splats = init_splats(scene_center, scene_radius, n_init, device, init_ply=init_ply)
    optimizers = make_optimizers(splats, scene_radius)

    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=2e-4,
        refine_start_iter=500,
        refine_stop_iter=int(n_iters * 0.75),
        reset_every=3000,
        refine_every=100,
        verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=float(scene_radius))

    K = torch.tensor(make_K(view_size, fov_deg), device=device, dtype=torch.float32)
    H = W = view_size

    # Pre-upload images / viewmats to GPU as a list (lighter than one big tensor)
    gts = [torch.tensor(v["image"], device=device) for v in views]
    vms = [torch.tensor(v["viewmat"], device=device) for v in views]
    print(f"[recon] training {n_iters} iters · {len(views)} views · {view_size}×{view_size} @ {fov_deg}° fov")

    t0 = time.time()
    for step in range(n_iters):
        idx = int(torch.randint(0, len(views), (1,)).item())
        gt = gts[idx]; viewmat = vms[idx]
        img, info = render_view(splats, viewmat, K, W, H)

        loss_l1 = (img - gt).abs().mean()
        if ssim_weight > 0:
            loss_ssim = 1.0 - ssim_1d(img, gt)
            loss = (1.0 - ssim_weight) * loss_l1 + ssim_weight * loss_ssim
        else:
            loss = loss_l1

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        # gsplat strategy needs pre-backward hook
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)

        if step % log_every == 0 or step == n_iters - 1:
            dt = time.time() - t0
            n = splats["means"].shape[0]
            print(f"  iter {step:>5}/{n_iters}  loss={loss.item():.4f}  "
                  f"N={n:>7}  ({dt:.1f}s elapsed)")

    print(f"[recon] training done in {time.time()-t0:.1f}s · final N={splats['means'].shape[0]:,}")
    save_ply(splats, out_ply)


# ── PLY export (canonical 3DGS schema) ────────────────────────────
def save_ply(splats, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    means = splats["means"].detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = F.normalize(splats["quats"], dim=-1).detach().cpu().numpy()
    opac  = splats["opacities"].detach().cpu().numpy()
    rgb   = torch.sigmoid(splats["colors"]).detach().cpu().numpy()

    f_dc = (rgb - 0.5) / SH_C0

    N = means.shape[0]
    dtype = [
        ("x","f4"),("y","f4"),("z","f4"),
        ("nx","f4"),("ny","f4"),("nz","f4"),
        ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
        ("opacity","f4"),
        ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
        ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
    ]
    arr = np.zeros(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    arr["opacity"] = opac
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    print(f"[recon] wrote {path}  ({path.stat().st_size/1024/1024:.1f} MB · {N:,} gaussians)")


# ── main ──────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True,
                   help="scene dir, e.g. MultiPano/input/witch_hat_atelier")
    p.add_argument("--out",   default=None,
                   help="output PLY path (default: <scene>/scene.ply)")
    p.add_argument("--num-yaw",   type=int, default=8,   help="perspective views around (yaw)")
    p.add_argument("--num-pitch", type=int, default=3,   help="perspective views in pitch (-25..+25 deg)")
    p.add_argument("--fov",       type=float, default=90, help="perspective view FOV in degrees")
    p.add_argument("--view-size", type=int,   default=512, help="square perspective view resolution")
    p.add_argument("--camera-height", type=float, default=1.6, help="metres above ground for cameras")
    p.add_argument("--iters",     type=int,   default=10000)
    p.add_argument("--n-init",    type=int,   default=30000)
    p.add_argument("--init-ply",  default=None,
                   help="sparse-init from this PLY (e.g. depth-fusion scene.ply). "
                        "Greatly improves convergence vs random init.")
    p.add_argument("--ssim",      type=float, default=0.2,
                   help="weight of (1 - SSIM) in the loss (standard 3DGS uses 0.2)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA")
    device = torch.device("cuda:0")

    scene_dir = Path(args.scene)
    out_ply = Path(args.out) if args.out else scene_dir / "scene.ply"

    views, pin_pos, _scale = build_views(
        scene_dir, num_yaw=args.num_yaw, num_pitch=args.num_pitch,
        fov_deg=args.fov, view_size=args.view_size,
        camera_height=args.camera_height,
    )

    # Scene bounds from camera positions
    pp = np.stack(list(pin_pos.values()))
    scene_center = pp.mean(axis=0)
    scene_radius = float(np.linalg.norm(pp - scene_center, axis=1).max() + 4.0)
    print(f"[recon] scene centre = ({scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f})"
          f"  radius ≈ {scene_radius:.2f} m")

    train(views, scene_center, scene_radius, n_init=args.n_init, n_iters=args.iters,
          fov_deg=args.fov, view_size=args.view_size,
          out_ply=out_ply, device=device,
          init_ply=Path(args.init_ply) if args.init_ply else None,
          ssim_weight=args.ssim)


if __name__ == "__main__":
    main()
