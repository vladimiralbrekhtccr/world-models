"""Build a single Lyra-2 input where the camera SPINS IN PLACE 360°.

Frame 0 = looking north (chapel). Frames 0..N sweep yaw 0..360° (a small
helical dolly inward+outward by ±0.3 m is added so VIPE has some
parallax to anchor depth — pure rotation is out-of-distribution).

Output bundle (id=0):
  lyra_in_orbit/0.png  ← chapel first frame
  lyra_in_orbit/0.npz  ← 161-frame w2c with yaw sweep
  lyra_in_orbit/0.json ← 2 per-chunk captions
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from build_lyra_inputs import equirect_to_perspective


HERE = Path(__file__).parent
PANO = HERE / "input" / "witch_hat_atelier" / "pano_2.png"
OUT = HERE / "lyra_in_orbit"
OUT.mkdir(exist_ok=True)

H, W = 480, 832
HFOV_DEG = 70.0
NUM_FRAMES = 161
PARALLAX_RADIUS = 0.3   # very small inward/outward wobble for VIPE


def main():
    pano = np.array(Image.open(PANO).convert("RGB"))
    first = equirect_to_perspective(pano, H, W, HFOV_DEG, yaw_deg=0.0, pitch_deg=0.0)
    Image.fromarray(first).save(OUT / "0.png")

    ts = np.linspace(0, 1, NUM_FRAMES)
    yaws = 2 * math.pi * ts  # 0 → 2π
    # tiny helical wobble: small forward dolly that breathes in/out
    fwd = PARALLAX_RADIUS * np.sin(4 * math.pi * ts)  # ±0.3 m, 2 cycles

    w2c = np.zeros((NUM_FRAMES, 4, 4), dtype=np.float32)
    for i, (yaw, dz) in enumerate(zip(yaws, fwd)):
        c, s = math.cos(yaw), math.sin(yaw)
        # c2w rotation: camera +Z aligned with world (sin yaw, 0, cos yaw)
        # Equivalent to Ry(yaw) in active sense.
        R_c2w = np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c],
        ], dtype=np.float32)
        # camera position: small forward dolly in camera-local +Z, mapped to world
        cam_pos_world = R_c2w @ np.array([0, 0, dz], dtype=np.float32)
        # w2c = [R^T | -R^T t]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ cam_pos_world
        w2c[i, :3, :3] = R_w2c
        w2c[i, :3, 3] = t_w2c
        w2c[i, 3, 3] = 1.0

    fx = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float32)
    intr = np.tile(K, (NUM_FRAMES, 1, 1))

    np.savez(
        OUT / "0.npz",
        w2c=w2c, intrinsics=intr,
        image_height=np.int32(H), image_width=np.int32(W),
    )

    captions = {
        "0": "An anime-style stone chapel in a forest clearing, watercolor "
             "painterly style, soft dappled sunlight. The camera turns slowly "
             "to the right, panning across the clearing.",
        "81": "Continuing the slow rightward pan, sweeping across the forest "
              "surrounding the chapel — trees, ferns, mossy stones, light "
              "shafts — same Witch Hat Atelier watercolor style.",
    }
    (OUT / "0.json").write_text(json.dumps(captions, indent=2))
    print(f"wrote {OUT}/0.{{png,npz,json}}  ({NUM_FRAMES} frames, yaw 0→360°, ±{PARALLAX_RADIUS}m wobble)")


if __name__ == "__main__":
    main()
