#!/usr/bin/env python3
"""Diffusion-prior refinement of a 3DGS scene.

Plan A from RESEARCH-DIFFUSION-INTEGRATION.md, simpler-than-true-SDS variant:
1. Load an existing scene.ply (depth-fusion or coarse multi-view).
2. Sample N novel camera viewpoints around the scene.
3. Render each viewpoint via gsplat → rough_image.
4. Pass rough_image through SD 1.5 img2img at low strength (0.3-0.4) → cleaned_image.
   Diffusion smooths cross-view inconsistencies into a single coherent appearance.
5. Continue training gsplat with the cleaned_images as supervision.
6. Save refined PLY.

This is iterative diffusion-guided refinement (img2img-as-prior), not true SDS
(score-distillation). True SDS computes diffusion gradients through the
renderer; this just uses the diffusion output as a denoised "pseudo-GT".
Simpler, faster to prototype, qualitatively similar effect for our case.
"""
from __future__ import annotations

import argparse, math, sys, time
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


# ── PLY I/O ───────────────────────────────────────────────────────
def load_ply_to_params(path: Path, device: torch.device) -> nn.ParameterDict:
    ply = PlyData.read(str(path)); v = ply["vertex"]
    pos = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], -1).astype(np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1).astype(np.float32)
    opac = v["opacity"].astype(np.float32)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1).astype(np.float32)
    # Recover pre-sigmoid color from f_dc: rgb = sigmoid(stored), so stored = logit(rgb)
    rgb = np.clip(fdc * SH_C0 + 0.5, 1e-3, 1 - 1e-3)
    logit_rgb = np.log(rgb / (1 - rgb))
    N = len(pos)
    print(f"[load] {N:,} gaussians from {path.name}")
    return nn.ParameterDict({
        "means":     nn.Parameter(torch.tensor(pos,        dtype=torch.float32, device=device)),
        "quats":     nn.Parameter(torch.tensor(quats,      dtype=torch.float32, device=device)),
        "scales":    nn.Parameter(torch.tensor(scales,     dtype=torch.float32, device=device)),
        "opacities": nn.Parameter(torch.tensor(opac,       dtype=torch.float32, device=device)),
        "colors":    nn.Parameter(torch.tensor(logit_rgb,  dtype=torch.float32, device=device)),
    })


def save_ply(splats: nn.ParameterDict, path: Path):
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
    arr["x"], arr["y"], arr["z"] = means[:,0], means[:,1], means[:,2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:,0], f_dc[:,1], f_dc[:,2]
    arr["opacity"] = opac
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:,0], scales[:,1], scales[:,2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    print(f"[save] {path}  ({path.stat().st_size/1024/1024:.1f} MB · {N:,} gaussians)")


# ── camera helpers ────────────────────────────────────────────────
def look_at_R(yaw: float, pitch: float):
    fx, fy, fz = math.sin(yaw)*math.cos(pitch), math.sin(pitch), -math.cos(yaw)*math.cos(pitch)
    forward = np.array([fx, fy, fz], dtype=np.float64)
    right = np.cross(forward, [0,1,0]); right /= np.linalg.norm(right) + 1e-9
    cam_down = np.cross(right, forward)
    return np.stack([right, cam_down, forward], axis=0).astype(np.float32)


def make_viewmat(eye, yaw, pitch):
    R = look_at_R(yaw, pitch)
    t = -R @ np.asarray(eye, dtype=np.float64)
    M = np.eye(4, dtype=np.float32); M[:3,:3] = R; M[:3,3] = t.astype(np.float32)
    return M


def make_K(size, fov_deg):
    f = (size/2) / math.tan(math.radians(fov_deg)/2)
    return np.array([[f,0,size/2],[0,f,size/2],[0,0,1]], dtype=np.float32)


def render(splats, viewmat_t, K_t, W, H):
    quats_n = F.normalize(splats["quats"], dim=-1)
    img, _, info = gsplat.rasterization(
        splats["means"], quats_n,
        torch.exp(splats["scales"]),
        torch.sigmoid(splats["opacities"]),
        torch.sigmoid(splats["colors"]),
        viewmat_t[None], K_t[None], W, H,
        packed=False, render_mode="RGB",
    )
    return img.squeeze(0), info


# ── novel-viewpoint sampling ──────────────────────────────────────
def sample_novel_view(scene_center, scene_radius, rng):
    """Pick a camera position somewhere inside the scene, looking inward."""
    # Random direction from centre, mostly horizontal
    theta = rng.uniform(0, 2*math.pi)
    radial = rng.uniform(0.2, 0.7) * scene_radius   # not too close, not too far
    height = scene_center[1] + rng.uniform(-0.5, 2.0)
    eye = np.array([
        scene_center[0] + radial * math.cos(theta),
        height,
        scene_center[2] + radial * math.sin(theta),
    ], dtype=np.float32)
    # Yaw to look back toward centre (with small jitter)
    delta = scene_center - eye
    yaw = math.atan2(delta[0], -delta[2]) + rng.uniform(-0.4, 0.4)
    pitch = rng.uniform(-0.1, 0.15)
    return eye, yaw, pitch


# ── main loop ─────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-ply",  required=True)
    p.add_argument("--out-ply", required=True)
    p.add_argument("--diffusion-model", default="runwayml/stable-diffusion-v1-5",
                   help="HF id; SD 1.5 is the lightest sensible prior")
    p.add_argument("--strength", type=float, default=0.35,
                   help="img2img strength; lower = preserve more structure")
    p.add_argument("--scene-prompt", default="a small stone chapel in a forest clearing, watercolor illustration, Witch Hat Atelier style, soft daylight",
                   help="text prompt for the diffusion model")
    p.add_argument("--neg-prompt", default="blurry, smeared, distorted, mush",
                   help="negative prompt")
    p.add_argument("--n-novel-views", type=int, default=30, help="how many cleaned pseudo-GT views to use")
    p.add_argument("--view-size", type=int, default=384)
    p.add_argument("--fov", type=float, default=80)
    p.add_argument("--n-iters", type=int, default=2000, help="gsplat refinement iters")
    p.add_argument("--guidance-scale", type=float, default=7.0)
    args = p.parse_args()

    dev = torch.device("cuda:0")
    rng = np.random.default_rng(42)

    # 1. Load splats
    splats = load_ply_to_params(Path(args.in_ply), dev)

    # 2. Compute scene bounds for novel-view sampling
    means = splats["means"].detach().cpu().numpy()
    scene_center = means.mean(axis=0)
    scene_radius = float(np.linalg.norm(means - scene_center, axis=1).mean())
    print(f"[scene] centre ({scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f})  radius ≈ {scene_radius:.2f} m")

    # 3. Load diffusion pipeline (img2img)
    print(f"[diff] loading {args.diffusion_model} (img2img) …")
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.diffusion_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(dev)
    pipe.set_progress_bar_config(disable=True)
    print(f"[diff] ready")

    # 4. Generate cleaned pseudo-GT views once (cache them; reuse across iters)
    print(f"[gen] rendering + cleaning {args.n_novel_views} novel views…")
    K = torch.tensor(make_K(args.view_size, args.fov), device=dev, dtype=torch.float32)
    pseudo_gts = []   # list of (cleaned_image_HxWx3_float32, viewmat_np)
    for k in range(args.n_novel_views):
        eye, yaw, pitch = sample_novel_view(scene_center, scene_radius, rng)
        viewmat = make_viewmat(eye, yaw, pitch)
        with torch.no_grad():
            rough, _ = render(splats,
                              torch.tensor(viewmat, device=dev, dtype=torch.float32),
                              K, args.view_size, args.view_size)
        rough_np = (rough.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        rough_pil = Image.fromarray(rough_np)
        # img2img — strength below 0.5 preserves geometry, just denoises
        cleaned = pipe(
            prompt=args.scene_prompt,
            negative_prompt=args.neg_prompt,
            image=rough_pil,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=20,
        ).images[0]
        cleaned_np = np.asarray(cleaned).astype(np.float32) / 255.0
        pseudo_gts.append((torch.tensor(cleaned_np, device=dev), torch.tensor(viewmat, device=dev, dtype=torch.float32)))
        if k % 5 == 0:
            print(f"  [{k+1}/{args.n_novel_views}]  eye=({eye[0]:.1f},{eye[2]:.1f}) yaw={math.degrees(yaw):.0f}°")

    # Free pipeline memory before training
    del pipe
    torch.cuda.empty_cache()
    print(f"[gen] {len(pseudo_gts)} pseudo-GT views ready · pipeline freed")

    # 5. Refine — train gsplat against the cleaned pseudo-GTs
    optimizers = {
        "means":     torch.optim.Adam([splats["means"]],     lr=1e-4 * scene_radius),
        "scales":    torch.optim.Adam([splats["scales"]],    lr=2e-3),
        "quats":     torch.optim.Adam([splats["quats"]],     lr=5e-4),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=2e-2),
        "colors":    torch.optim.Adam([splats["colors"]],    lr=1.5e-3),
    }
    strategy = DefaultStrategy(
        prune_opa=0.005, grow_grad2d=2e-4,
        refine_start_iter=200, refine_stop_iter=int(args.n_iters * 0.7),
        reset_every=2000, refine_every=100, verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=float(scene_radius))

    print(f"[refine] {args.n_iters} iters against {len(pseudo_gts)} pseudo-GT views")
    t0 = time.time()
    for step in range(args.n_iters):
        idx = int(torch.randint(0, len(pseudo_gts), (1,)).item())
        gt, viewmat = pseudo_gts[idx]
        img, info = render(splats, viewmat, K, args.view_size, args.view_size)
        loss = (img - gt).abs().mean()
        for opt in optimizers.values(): opt.zero_grad(set_to_none=True)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values(): opt.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        if step % 200 == 0:
            print(f"  iter {step:>4}/{args.n_iters}  loss={loss.item():.4f}  N={splats['means'].shape[0]:>7}  ({time.time()-t0:.1f}s)")

    print(f"[refine] done in {time.time()-t0:.1f}s · final N={splats['means'].shape[0]:,}")
    save_ply(splats, Path(args.out_ply))


if __name__ == "__main__":
    main()
