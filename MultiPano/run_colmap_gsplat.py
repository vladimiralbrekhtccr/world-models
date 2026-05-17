"""End-to-end: video → frames → COLMAP SfM → gsplat training → PLY.

This is the classic 3DGS pipeline nerfstudio's splatfacto wraps:
  1. extract N frames from the video
  2. COLMAP (via pycolmap): SIFT features → sequential matching → mapper
  3. gsplat: init gaussians from sparse points (colors from COLMAP), train
     with photometric L1 + SSIM, adaptive density control (DefaultStrategy),
     save standard 3DGS PLY

Tuned for short hand-held phone videos. ~10-30 min on one H100.
"""
import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

import gsplat
from gsplat import DefaultStrategy

SH_C0 = 0.28209479177387814


# ---------- step 1: extract frames ----------
def extract_frames(video: Path, image_dir: Path, n_frames: int, max_side: int):
    image_dir.mkdir(parents=True, exist_ok=True)
    for old in image_dir.glob("*.png"):
        old.unlink()
    # Get total frame count + dims
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    w, h, total = [int(x) for x in probe.stdout.strip().split(",")]
    step = max(1, total // n_frames)
    scale_w = max_side if w >= h else int(round(max_side * w / h / 2)) * 2
    scale_h = max_side if h >= w else int(round(max_side * h / w / 2)) * 2
    vf = f"scale={scale_w if w >= h else -2}:{scale_h if h > w else -2},select='not(mod(n,{step}))'"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", vf, "-vsync", "vfr", "-frames:v", str(n_frames),
         str(image_dir / "frame_%04d.png")],
        check=True,
    )
    frames = sorted(image_dir.glob("*.png"))
    print(f"  extracted {len(frames)} frames at ~{scale_w}x{scale_h} → {image_dir}")
    return frames


# ---------- step 2: COLMAP SfM ----------
def run_colmap(image_dir: Path, work_dir: Path):
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    if db_path.exists():
        db_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True)

    print("  [colmap] SIFT extraction (boosted feature count)…")
    extr_opts = pycolmap.FeatureExtractionOptions()
    extr_opts.sift.max_num_features = 16384  # default 8192; bump for textureless rooms
    extr_opts.sift.peak_threshold = 0.001    # default 0.0067; lower = more features
    extr_opts.use_gpu = True
    pycolmap.extract_features(
        db_path, image_dir,
        extraction_options=extr_opts,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )

    print("  [colmap] exhaustive matching…")
    pycolmap.match_exhaustive(db_path)

    print("  [colmap] incremental mapping…")
    maps = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir)
    if not maps:
        raise RuntimeError("COLMAP found no reconstruction")
    rec = maps[0]
    print(f"  [colmap] {rec.num_reg_images()} registered, "
          f"{rec.num_points3D()} sparse points")
    return rec


# ---------- step 3: gsplat training ----------
def init_splats(rec, device):
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    cols = np.array([np.array(p.color) / 255.0 for p in rec.points3D.values()],
                    dtype=np.float32)
    cols = np.clip(cols, 1e-3, 1 - 1e-3)
    logit_rgb = np.log(cols / (1.0 - cols))
    N = pts.shape[0]
    print(f"  [init] {N:,} gaussians from COLMAP sparse cloud")

    means = torch.tensor(pts, dtype=torch.float32, device=device)
    # initial scale: 0.5% of scene radius
    scene_radius = float(np.linalg.norm(pts - pts.mean(0), axis=1).max())
    init_s = max(scene_radius * 0.002, 1e-3)
    scales = torch.full((N, 3), math.log(init_s), device=device)
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
    opacities = torch.full((N,), 0.0, device=device)  # logit(0.5)
    colors = torch.tensor(logit_rgb, dtype=torch.float32, device=device)

    splats = nn.ParameterDict({
        "means":     nn.Parameter(means),
        "scales":    nn.Parameter(scales),
        "quats":     nn.Parameter(quats),
        "opacities": nn.Parameter(opacities),
        "colors":    nn.Parameter(colors),
    })
    return splats, scene_radius


def build_views(rec, image_dir: Path, device):
    views = []
    for img_id in sorted(rec.reg_image_ids()):
        img = rec.images[img_id]
        cam = rec.cameras[img.camera_id]
        T34 = img.cam_from_world().matrix()  # OpenCV w2c (3,4)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :4] = T34
        # Intrinsics
        if cam.model == pycolmap.CameraModelId.SIMPLE_RADIAL:
            f, cx, cy, _k = cam.params
            fx = fy = f
        elif cam.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
            f, cx, cy = cam.params
            fx = fy = f
        elif cam.model == pycolmap.CameraModelId.PINHOLE:
            fx, fy, cx, cy = cam.params
        else:
            fx = cam.focal_length_x; fy = cam.focal_length_y
            cx = cam.principal_point_x; cy = cam.principal_point_y
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        pil = Image.open(image_dir / img.name).convert("RGB")
        if pil.size != (cam.width, cam.height):
            pil = pil.resize((cam.width, cam.height), Image.LANCZOS)
        arr = torch.from_numpy(np.array(pil)).to(device).float() / 255.0  # (H,W,3)
        views.append({
            "name": img.name,
            "image": arr, "viewmat": torch.from_numpy(viewmat).to(device),
            "K": torch.from_numpy(K).to(device),
            "H": cam.height, "W": cam.width,
        })
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
    device = splats["means"].device
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

    print(f"  [train] {iters} iters · {len(views)} views · scene_radius={scene_radius:.2f}")
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(iters):
        idx = int(rng.integers(0, len(views))); v = views[idx]
        quats_n = F.normalize(splats["quats"], dim=-1)
        scales_p = torch.exp(splats["scales"])
        opac_p = torch.sigmoid(splats["opacities"])
        colors_p = torch.sigmoid(splats["colors"])
        img, _, info = gsplat.rasterization(
            splats["means"], quats_n, scales_p, opac_p, colors_p,
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

    print(f"  [train] done in {time.time()-t0:.1f}s, final N={splats['means'].shape[0]:,}")
    save_ply(splats, out_ply)


def save_ply(splats, out_path: Path):
    means = splats["means"].detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opac = splats["opacities"].detach().cpu().numpy()
    # store color as SH_dc = (rgb - 0.5) / SH_C0  so standard viewers decode correctly
    rgb = torch.sigmoid(splats["colors"]).detach().cpu().numpy()
    rgb = np.clip(rgb, 1e-3, 1 - 1e-3)
    fdc = (rgb - 0.5) / SH_C0
    N = means.shape[0]
    dtype = [
        ("x","f4"),("y","f4"),("z","f4"),
        ("nx","f4"),("ny","f4"),("nz","f4"),
        ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
        ("opacity","f4"),
        ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
        ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
    ]
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
    print(f"  [ply] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, {N:,} gaussians)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--work_dir", type=Path, required=True)
    ap.add_argument("--out_ply", type=Path, required=True)
    ap.add_argument("--n_frames", type=int, default=80)
    ap.add_argument("--max_side", type=int, default=960)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--ssim_weight", type=float, default=0.2)
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--skip_colmap", action="store_true")
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.work_dir / "images"

    if not args.skip_extract:
        print("== step 1: extract frames ==")
        extract_frames(args.video, image_dir, args.n_frames, args.max_side)

    rec_path = args.work_dir / "sparse" / "0"
    if args.skip_colmap and rec_path.exists():
        print("== step 2: load existing COLMAP recon ==")
        rec = pycolmap.Reconstruction(str(rec_path))
    else:
        print("== step 2: COLMAP SfM ==")
        rec = run_colmap(image_dir, args.work_dir)

    print("== step 3: gsplat training ==")
    splats, scene_radius = init_splats(rec, "cuda")
    views = build_views(rec, image_dir, "cuda")
    train(splats, scene_radius, views, args.out_ply, args.iters, args.ssim_weight)


if __name__ == "__main__":
    main()
