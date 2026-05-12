#!/usr/bin/env python3
"""Render a short MP4 showing a cloud of 3D Gaussians converging from
random initialisation to a target shape, with the camera orbiting around.

Used by the /3dgs-explainer/ deployed at:
    https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs-explainer/

Each Gaussian is rendered as a 2D radial splat with alpha-falloff,
back-to-front-composited per frame. Output: out/training.mp4.

Run from this folder:
    source .venv/bin/activate
    python render_gaussian_video.py
"""
from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

# ── config ──────────────────────────────────────────────────────────
OUT     = Path(__file__).parent / "out" / "training.mp4"
WIDTH   = 720
HEIGHT  = 404                                    # must be even for yuv420p
FPS     = 30
SECONDS = 6
FRAMES  = FPS * SECONDS
N_GAUSS = 480
SEED    = 17

# ── target shape: a torus + a small cube to break the symmetry ─────
def sample_target(n: int, rng: np.random.Generator) -> np.ndarray:
    # ~75% torus around y=0, 25% cube floating to the side
    n_torus = int(n * 0.75)
    n_cube  = n - n_torus

    R, r = 1.6, 0.55
    u = rng.uniform(0, 2 * np.pi, n_torus)
    v = rng.uniform(0, 2 * np.pi, n_torus)
    tx = (R + r * np.cos(v)) * np.cos(u)
    ty = 0.55 * np.sin(v)
    tz = (R + r * np.cos(v)) * np.sin(u)
    torus = np.stack([tx, ty, tz], axis=-1)

    # Cube at offset, side 0.9
    cu = rng.uniform(-0.45, 0.45, (n_cube, 3)) + np.array([0.0, 0.95, 0.0])
    return np.concatenate([torus, cu], axis=0)


def hsl_rgb(h, s, l):
    return np.array(colorsys.hls_to_rgb(h, l, s))


def target_colors(positions: np.ndarray) -> np.ndarray:
    """Colour by azimuth angle (rainbow) for visual richness."""
    out = np.zeros((len(positions), 3))
    ang = np.arctan2(positions[:, 2], positions[:, 0])
    for i in range(len(positions)):
        h = (ang[i] + np.pi) / (2 * np.pi)            # [0, 1]
        # high y points get slightly warmer
        h = (h + 0.07 * positions[i, 1]) % 1.0
        out[i] = hsl_rgb(h, 0.65, 0.55)
    return out


# ── camera helpers ─────────────────────────────────────────────────
def lookat_basis(eye: np.ndarray, target=np.zeros(3), world_up=np.array([0, 1, 0])):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, world_up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f], axis=0)  # rows = camera axes


def project(points: np.ndarray, R: np.ndarray, eye: np.ndarray,
            W: int, H: int, fov_y: float = 1.05):
    cam = (points - eye) @ R.T              # (N, 3) in camera space
    z = cam[:, 2]
    safe = z > 0.05
    f = H / (2 * np.tan(fov_y / 2))
    px = np.where(safe, (cam[:, 0] / np.maximum(z, 0.05)) * f + W / 2, -1)
    py = np.where(safe, -(cam[:, 1] / np.maximum(z, 0.05)) * f + H / 2, -1)
    return np.stack([px, py], axis=-1), z, safe


# ── render ─────────────────────────────────────────────────────────
def smoothstep(t):
    return t * t * (3 - 2 * t)


def render_frame(canvas: np.ndarray,
                 pixels: np.ndarray, depths: np.ndarray, safe: np.ndarray,
                 colours: np.ndarray, alphas: np.ndarray, sizes: np.ndarray,
                 W: int, H: int):
    # Sort back-to-front (large z first)
    order = np.argsort(-depths)
    for i in order:
        if not safe[i]:
            continue
        cx, cy = pixels[i]
        r = sizes[i]
        sz = int(np.clip(r * 3.5, 5, 80))
        x0, x1 = max(0, int(cx - sz)), min(W, int(cx + sz + 1))
        y0, y1 = max(0, int(cy - sz)), min(H, int(cy + sz + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        dx = xs - cx
        dy = ys - cy
        density = np.exp(-0.5 * (dx * dx + dy * dy) / (r * r))
        a = density * alphas[i]
        a = a[..., None]
        col = colours[i].reshape(1, 1, 3)
        region = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = region * (1 - a) + col * a


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    target = sample_target(N_GAUSS, rng)
    # init: random in a 4m sphere
    direc = rng.standard_normal((N_GAUSS, 3))
    direc /= np.linalg.norm(direc, axis=1, keepdims=True) + 1e-9
    radii = rng.uniform(0.3, 2.5, N_GAUSS)
    init = direc * radii[:, None]

    target_col = target_colors(target)
    init_col   = np.full_like(target_col, 0.45)   # warm grey

    # Initial sizes are LARGER and shrink as training "converges".
    base_size = rng.uniform(8, 14, N_GAUSS)        # pixel radius at unit depth

    print(f"[render] {FRAMES} frames @ {FPS} fps · {N_GAUSS} gaussians · {WIDTH}x{HEIGHT}")
    writer = imageio.get_writer(
        OUT, fps=FPS, codec="libx264", quality=8,
        macro_block_size=1, output_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for f in range(FRAMES):
            t_norm = f / max(FRAMES - 1, 1)
            s = smoothstep(t_norm)

            # Most of the convergence happens in the first 70% of the clip,
            # then the camera keeps orbiting on a settled scene.
            conv = smoothstep(min(1.0, t_norm / 0.75))
            pos = init * (1 - conv) + target * conv
            col = init_col * (1 - conv) + target_col * conv
            alpha = 0.45 + 0.35 * conv          # blobs become more opaque
            size_scale = 1.0 - 0.55 * conv      # and smaller as they tighten

            # Camera: 1.25 turns around y axis, slight up tilt
            ang = t_norm * 2 * np.pi * 1.25
            cam_dist = 5.4
            eye = np.array([np.cos(ang) * cam_dist, 1.6, np.sin(ang) * cam_dist])
            R = lookat_basis(eye)

            pixels, depths, safe = project(pos, R, eye, WIDTH, HEIGHT)
            # 2D splat size: bigger when closer
            sizes = (1.0 / np.maximum(depths, 0.05)) * base_size * size_scale
            sizes = np.clip(sizes, 1.5, 40.0)

            canvas = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
            canvas[..., 0] = 12 / 255
            canvas[..., 1] = 14 / 255
            canvas[..., 2] = 22 / 255

            alphas = np.full(N_GAUSS, alpha)
            render_frame(canvas, pixels, depths, safe, col, alphas, sizes, WIDTH, HEIGHT)
            frame = (canvas * 255).clip(0, 255).astype(np.uint8)
            writer.append_data(frame)
            if f % 30 == 0:
                print(f"  frame {f:>4}/{FRAMES}  t={t_norm:.2f}", flush=True)
    finally:
        writer.close()

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"[render] wrote {OUT}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
