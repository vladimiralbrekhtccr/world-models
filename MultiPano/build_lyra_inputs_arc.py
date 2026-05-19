"""Build a Lyra-2 input where the camera ORBITS AROUND the chapel.

Unlike the in-place spin (lyra_in_orbit), here the camera physically
circles a point ahead of it — real translation → parallax → usable depth
for 3DGS. 481 frames (Lyra's max) for dense view coverage.

Frame 0: in front of the chapel, looking at it (matches the pano slice).
Then the camera swings a full circle around the chapel, always aimed at it,
with a gentle height bob.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from build_lyra_inputs import equirect_to_perspective


HERE = Path(__file__).parent
PANO = HERE / "input" / "witch_hat_atelier" / "pano_2.png"
OUT = HERE / "lyra_in_arc"
OUT.mkdir(exist_ok=True)

H, W = 480, 832
HFOV_DEG = 70.0
NUM_FRAMES = 481            # Lyra max
CHAPEL_AHEAD = 9.0          # chapel sits ~9 units in front of frame-0 camera
ORBIT_RADIUS = 3.5          # camera circles the chapel at this radius
HEIGHT_BOB = 0.5            # gentle vertical oscillation


def main():
    pano = np.array(Image.open(PANO).convert("RGB"))
    first = equirect_to_perspective(pano, H, W, HFOV_DEG, yaw_deg=0.0, pitch_deg=0.0)
    Image.fromarray(first).save(OUT / "0.png")

    chapel = np.array([0.0, 0.0, CHAPEL_AHEAD], dtype=np.float64)
    ts = np.linspace(0.0, 1.0, NUM_FRAMES)

    w2c = np.zeros((NUM_FRAMES, 4, 4), dtype=np.float32)
    for i, t in enumerate(ts):
        theta = 2.0 * math.pi * t
        # camera circles the chapel; theta=0 → in front (z = CHAPEL_AHEAD - R)
        cam = chapel + ORBIT_RADIUS * np.array(
            [math.sin(theta), 0.0, -math.cos(theta)], dtype=np.float64)
        cam[1] += HEIGHT_BOB * math.sin(2.0 * theta)   # bob up/down

        # camera looks AT the chapel
        fwd = chapel - cam
        fwd /= np.linalg.norm(fwd)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up); right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        # camera-to-world rotation: columns = right, up(down for OpenCV), fwd
        R_c2w = np.stack([right, -cam_up, fwd], axis=1).astype(np.float64)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ cam
        w2c[i, :3, :3] = R_w2c
        w2c[i, :3, 3] = t_w2c
        w2c[i, 3, 3] = 1.0

    fx = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float32)
    intr = np.tile(K, (NUM_FRAMES, 1, 1))
    np.savez(OUT / "0.npz", w2c=w2c, intrinsics=intr,
             image_height=np.int32(H), image_width=np.int32(W))

    # captions keyed by chunk start (chunks of 81 frames)
    base = ("An anime-style stone chapel in a forest clearing, watercolor "
            "painterly style, soft dappled sunlight.")
    caps = {str(k): base + " The camera circles around the chapel, "
                            "revealing it from every side."
            for k in range(0, NUM_FRAMES, 81)}
    (OUT / "0.json").write_text(json.dumps(caps, indent=2))
    print(f"wrote {OUT}/0.{{png,npz,json}}  ({NUM_FRAMES} frames, "
          f"orbit R={ORBIT_RADIUS} around chapel @ {CHAPEL_AHEAD})")


if __name__ == "__main__":
    main()
