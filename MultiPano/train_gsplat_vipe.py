"""Train gsplat on the room video using VIPE poses (from Lyra recon) +
Lyra's PLY as initialization, with photometric L1 + SSIM and DefaultStrategy.

The point: VIPE already nailed pose estimation on this video (where COLMAP
failed). Use those poses + Lyra's PLY as init, then run real photometric
optimization with adaptive density control — should fix Lyra's pixel-aligned
ghosting if the per-frame supervision is consistent.
"""
import argparse
import math
import subprocess
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

SH_C0 = 0.28209479177387814


def extract_frames(video: Path, out_dir: Path, n_expected: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
         "-vsync", "0", str(out_dir / "frame_%04d.png")],
        check=True,
    )
    frames = sorted(out_dir.glob("*.png"))
    print(f"  extracted {len(frames)} frames → {out_dir}")
    if len(frames) != n_expected:
        print(f"  WARN: expected {n_expected}, got {len(frames)}; using min")
    return frames


def init_from_ply(ply_path: Path, device, min_opacity: float, max_radius: float, target: int):
    print(f"  init from {ply_path}")
    v = PlyData.read(str(ply_path))["vertex"].data
    n = len(v)
    op = 1.0 / (1.0 + np.exp(-v["opacity"]))
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    r = np.linalg.norm(xyz, axis=1)
    mask = (op > min_opacity) & (r < max_radius)
    idx = np.where(mask)[0]
    if len(idx) > target:
        idx = np.random.default_rng(0).choice(idx, target, replace=False)
    v = v[idx]
    N = len(v)
    print(f"  kept {N:,} gaussians (opacity>{min_opacity}, r<{max_radius})")

    means = torch.tensor(np.stack([v["x"], v["y"], v["z"]], axis=1), dtype=torch.float32, device=device)
    quats = torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1), dtype=torch.float32, device=device)
    scales = torch.tensor(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1), dtype=torch.float32, device=device)
    opacities = torch.tensor(v["opacity"], dtype=torch.float32, device=device)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip(fdc * SH_C0 + 0.5, 1e-3, 1 - 1e-3)
    colors = torch.tensor(np.log(rgb / (1 - rgb)), dtype=torch.float32, device=device)

    splats = nn.ParameterDict({
        "means":     nn.Parameter(means),
        "scales":    nn.Parameter(scales),
        "quats":     nn.Parameter(quats),
        "opacities": nn.Parameter(opacities),
        "colors":    nn.Parameter(colors),
    })
    scene_radius = float(np.percentile(np.linalg.norm(means.cpu().numpy() - means.cpu().numpy().mean(0), axis=1), 95))
    return splats, scene_radius


def load_views(frame_paths, cams_npz: Path, device):
    d = np.load(cams_npz)
    w2c = d["w2c_vipe"].astype(np.float32)
    K = d["intrinsics_vipe"].astype(np.float32)
    n = min(len(frame_paths), w2c.shape[0])
    views = []
    for i in range(n):
        pil = Image.open(frame_paths[i]).convert("RGB")
        W, H = pil.size
        arr = torch.from_numpy(np.array(pil)).to(device).float() / 255.0
        views.append({
            "image": arr,
            "viewmat": torch.from_numpy(w2c[i]).to(device),
            "K": torch.from_numpy(K[i]).to(device),
            "W": W, "H": H,
        })
    print(f"  loaded {len(views)} views at {views[0]['W']}x{views[0]['H']}")
    return views


def ssim_1d(x, y, window_size=11):
    x = x.permute(2, 0, 1).unsqueeze(0); y = y.permute(2, 0, 1).unsqueeze(0)
    pad = window_size // 2
    coords = torch.arange(window_size, device=x.device, dtype=torch.float32) - pad
    g = torch.exp(-(coords**2) / (2 * 1.5**2)); g = (g / g.sum()).view(1, 1, 1, -1)
    g = g.expand(3, 1, 1, window_size); gT = g.transpose(2, 3)
    def conv(t):
        t = F.conv2d(t, g,  padding=(0, pad), groups=3)
        t = F.conv2d(t, gT, padding=(pad, 0), groups=3); return t
    mu_x, mu_y = conv(x), conv(y); mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sig_x2 = conv(x*x) - mu_x2; sig_y2 = conv(y*y) - mu_y2; sig_xy = conv(x*y) - mu_xy
    C1, C2 = 0.01**2, 0.03**2
    return (((2*mu_xy + C1)*(2*sig_xy + C2)) /
            ((mu_x2 + mu_y2 + C1)*(sig_x2 + sig_y2 + C2))).mean()


def train(splats, scene_radius, views, out_ply: Path, iters: int, ssim_weight: float):
    optimizers = {
        "means":     torch.optim.Adam([splats["means"]],     lr=1.6e-4 * scene_radius),
        "scales":    torch.optim.Adam([splats["scales"]],    lr=5e-3),
        "quats":     torch.optim.Adam([splats["quats"]],     lr=1e-3),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=5e-2),
        "colors":    torch.optim.Adam([splats["colors"]],    lr=2.5e-3),
    }
    strategy = DefaultStrategy(
        prune_opa=0.005, grow_grad2d=2e-4,
        refine_start_iter=500, refine_stop_iter=int(iters * 0.85),
        reset_every=3000, refine_every=100, verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=float(scene_radius))

    print(f"  training {iters} iters · {len(views)} views · scene_radius~{scene_radius:.2f}")
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(iters):
        v = views[int(rng.integers(0, len(views)))]
        quats_n = F.normalize(splats["quats"], dim=-1)
        img, _, info = gsplat.rasterization(
            splats["means"], quats_n,
            torch.exp(splats["scales"]),
            torch.sigmoid(splats["opacities"]),
            torch.sigmoid(splats["colors"]),
            v["viewmat"][None], v["K"][None], v["W"], v["H"],
            packed=False, render_mode="RGB",
        )
        gt = v["image"]; pred = img[0]
        loss_l1 = (pred - gt).abs().mean()
        if ssim_weight > 0:
            loss = (1 - ssim_weight) * loss_l1 + ssim_weight * (1 - ssim_1d(pred, gt))
        else:
            loss = loss_l1
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        if step % 250 == 0 or step == iters - 1:
            print(f"    iter {step:5d}/{iters}  loss={loss.item():.4f}  "
                  f"N={splats['means'].shape[0]:>7,d}  ({time.time()-t0:.1f}s)")

    print(f"  done in {time.time()-t0:.1f}s, final N={splats['means'].shape[0]:,}")
    save_ply(splats, out_ply)


def save_ply(splats, out_path: Path):
    means = splats["means"].detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opac = splats["opacities"].detach().cpu().numpy()
    rgb = torch.sigmoid(splats["colors"]).detach().cpu().numpy()
    rgb = np.clip(rgb, 1e-3, 1 - 1e-3)
    fdc = (rgb - 0.5) / SH_C0
    N = means.shape[0]
    dtype = [("x","f4"),("y","f4"),("z","f4"),("nx","f4"),("ny","f4"),("nz","f4"),
             ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),("opacity","f4"),
             ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
             ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4")]
    arr = np.empty(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means[:,0], means[:,1], means[:,2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = fdc[:,0], fdc[:,1], fdc[:,2]
    arr["opacity"] = opac
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:,0], scales[:,1], scales[:,2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    el = PlyElement.describe(arr, "vertex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el], text=False).write(str(out_path))
    print(f"  wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, {N:,} gaussians)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="Video matching VIPE poses (resolution must match)")
    ap.add_argument("--cameras_npz", type=Path, required=True)
    ap.add_argument("--init_ply", type=Path, required=True, help="Init from this PLY (e.g. Lyra's)")
    ap.add_argument("--frames_dir", type=Path, required=True)
    ap.add_argument("--out_ply", type=Path, required=True)
    ap.add_argument("--init_min_opacity", type=float, default=0.1)
    ap.add_argument("--init_max_radius", type=float, default=20.0)
    ap.add_argument("--init_target", type=int, default=400_000)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--ssim_weight", type=float, default=0.2)
    args = ap.parse_args()

    print("== extract frames ==")
    frames = extract_frames(args.video, args.frames_dir, 160)
    print("== load views ==")
    views = load_views(frames, args.cameras_npz, "cuda")
    print("== init splats ==")
    splats, scene_radius = init_from_ply(
        args.init_ply, "cuda",
        args.init_min_opacity, args.init_max_radius, args.init_target,
    )
    print("== train ==")
    train(splats, scene_radius, views, args.out_ply, args.iters, args.ssim_weight)


if __name__ == "__main__":
    main()
