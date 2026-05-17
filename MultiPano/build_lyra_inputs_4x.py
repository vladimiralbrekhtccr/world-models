"""Build 4 Lyra-2 input bundles from pano_2, looking N / E / S / W.

Outputs under MultiPano/lyra_in_4x/:
  0.png  0.npz  0.json   ← yaw 0   (north, chapel)
  1.png  1.npz  1.json   ← yaw 90  (east)
  2.png  2.npz  2.json   ← yaw 180 (south)
  3.png  3.npz  3.json   ← yaw 270 (west)
  yaws.json              ← maps id → yaw_deg for the merger

Each first frame is sliced from pano_2 at the given yaw; each trajectory is
a forward-dolly in *that* camera's local +Z (i.e. world direction = yaw).
All four trajectories start at world origin (pin 2's POV).
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from build_lyra_inputs import equirect_to_perspective  # reuse


HERE = Path(__file__).parent
PANO = HERE / "input" / "witch_hat_atelier" / "pano_2.png"
OUT = HERE / "lyra_in_4x"
OUT.mkdir(exist_ok=True)

H, W = 480, 832
HFOV_DEG = 70.0
NUM_FRAMES = 161
DOLLY_DIST = 8.0   # forward distance in Lyra-frame units (will be rescaled by pose_scale)

YAWS = [0, 90, 180, 270]  # degrees: N, E, S, W
CAPTIONS = {
    0: ("A small stone chapel in a forest clearing, watercolor anime style, "
        "soft light filtering through trees, Witch Hat Atelier."),
    1: ("Looking east through the forest from the chapel clearing, tall trees, "
        "dappled sunlight on mossy ground, watercolor anime style."),
    2: ("Looking south back along the path through the woods, weathered tree "
        "trunks and ferns, watercolor anime style, soft afternoon light."),
    3: ("Looking west into deeper forest, undergrowth and slanted shadows, "
        "watercolor anime style, dust motes in shafts of light."),
}


def make_trajectory(num_frames: int, h: int, w: int, hfov_deg: float, dist: float):
    """Forward dolly in OpenCV camera frame: camera moves +Z toward depth dist."""
    ts = np.linspace(0, 1, num_frames)
    ease = 0.5 - 0.5 * np.cos(math.pi * ts)
    cam_pos = np.zeros((num_frames, 3), dtype=np.float32)
    cam_pos[:, 2] = dist * ease  # +Z forward in camera frame

    w2c = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    w2c[:, :3, 3] = -cam_pos

    fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float32)
    intr = np.tile(K, (num_frames, 1, 1))
    return w2c.astype(np.float32), intr


def main():
    pano = np.array(Image.open(PANO).convert("RGB"))
    w2c, intr = make_trajectory(NUM_FRAMES, H, W, HFOV_DEG, DOLLY_DIST)

    for i, yaw in enumerate(YAWS):
        first = equirect_to_perspective(pano, H, W, HFOV_DEG,
                                        yaw_deg=float(yaw), pitch_deg=0.0)
        Image.fromarray(first).save(OUT / f"{i}.png")
        np.savez(
            OUT / f"{i}.npz",
            w2c=w2c, intrinsics=intr,
            image_height=np.int32(H), image_width=np.int32(W),
        )
        caps = {
            "0":  CAPTIONS[i],
            "81": CAPTIONS[i] + " The camera glides forward into the scene.",
        }
        (OUT / f"{i}.json").write_text(json.dumps(caps, indent=2))
        print(f"id {i}: yaw {yaw}°  →  {i}.png  {i}.npz  {i}.json")

    (OUT / "yaws.json").write_text(json.dumps(
        {str(i): yaw for i, yaw in enumerate(YAWS)}, indent=2))
    print(f"wrote {OUT}/yaws.json")


if __name__ == "__main__":
    main()
