"""Build Lyra-2 inputs from our pano_2 + poses.json.

Outputs (under MultiPano/lyra_in/):
  first_frame.png     — perspective slice of pano_2 (chapel-facing), 480x832
  trajectory.npz      — w2c (N,4,4), intrinsics (N,3,3), image_height, image_width
  captions.json       — {"0": "...", "81": "..."}  keyed by chunk-start frame
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).parent
PANO = HERE / "input" / "witch_hat_atelier" / "pano_2.png"
POSES = HERE / "input" / "witch_hat_atelier" / "poses.json"
OUT = HERE / "lyra_in"
OUT.mkdir(exist_ok=True)

H, W = 480, 832
HFOV_DEG = 70.0  # roughly matches Lyra's training distribution
NUM_FRAMES = 161  # = 2 AR chunks of 81 frames (with 1-frame overlap)
YAW_FACING = 0.0  # pano was shot looking north; +x = east, -z = north


def equirect_to_perspective(pano: np.ndarray, h: int, w: int, hfov_deg: float, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Bilinear-sample a perspective view out of a 2:1 equirectangular pano."""
    pano_h, pano_w = pano.shape[:2]
    hfov = math.radians(hfov_deg)
    fx = (w / 2) / math.tan(hfov / 2)
    fy = fx  # square pixels
    cx, cy = w / 2, h / 2
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_cam = (xx - cx) / fx
    y_cam = (yy - cy) / fy
    z_cam = np.ones_like(x_cam)
    norm = np.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2)
    x_cam, y_cam, z_cam = x_cam / norm, y_cam / norm, z_cam / norm

    cp, sp = math.cos(pitch), math.sin(pitch)
    y_w = cp * y_cam - sp * z_cam
    z_w = sp * y_cam + cp * z_cam
    x_w = x_cam
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    x_world = cy_ * x_w + sy_ * z_w
    z_world = -sy_ * x_w + cy_ * z_w
    y_world = y_w

    lon = np.arctan2(x_world, z_world)
    lat = np.arcsin(np.clip(y_world, -1.0, 1.0))
    u = (lon / (2 * math.pi) + 0.5) * pano_w
    v = (lat / math.pi + 0.5) * pano_h

    u0 = np.floor(u).astype(int) % pano_w
    u1 = (u0 + 1) % pano_w
    v0 = np.clip(np.floor(v).astype(int), 0, pano_h - 1)
    v1 = np.clip(v0 + 1, 0, pano_h - 1)
    du = (u - np.floor(u))[..., None]
    dv = (v - np.floor(v))[..., None]

    p00 = pano[v0, u0].astype(np.float32)
    p01 = pano[v0, u1].astype(np.float32)
    p10 = pano[v1, u0].astype(np.float32)
    p11 = pano[v1, u1].astype(np.float32)
    out = (1 - du) * (1 - dv) * p00 + du * (1 - dv) * p01 + (1 - du) * dv * p10 + du * dv * p11
    return np.clip(out, 0, 255).astype(np.uint8)


def build_trajectory(num_frames: int, h: int, w: int, hfov_deg: float):
    """Smooth ease-in/ease-out dolly forward from pano-2 origin toward the chapel,
    with a small left-right sway. Camera always faces -Z (north).

    World frame: +X east, +Y up, +Z south (so camera looks -Z = north).
    pano_2 was shot at pin2; pin1 ~ 7m forward-left, pin3 ~ 7m forward-right.
    """
    poses = json.loads(POSES.read_text())
    cams = poses["cameras"]
    scale = poses["scale_m_per_px"]
    p2 = np.array([cams["2"]["px"], cams["2"]["py"]])
    p1 = np.array([cams["1"]["px"], cams["1"]["py"]])
    p3 = np.array([cams["3"]["px"], cams["3"]["py"]])

    # world(px-relative-to-pin2): +x_image=east, -y_image=north
    def to_world(pin):
        d = (pin - p2) * scale
        return np.array([float(d[0]), 0.0, float(-d[1])])  # x=east, y=up, z=south(+) so north=-z

    w1 = to_world(p1)
    w3 = to_world(p3)
    chapel_ahead = -((-w1[2] + -w3[2]) / 2)  # avg forward depth of pins ~ 12m north → z = -12
    chapel = np.array([0.0, 0.0, chapel_ahead * 0.7])  # somewhere ahead

    # Path: pin2 → 70 % dolly toward chapel, plus small left-right sway
    ts = np.linspace(0, 1, num_frames)
    ease = 0.5 - 0.5 * np.cos(math.pi * ts)  # ease in/out
    forward = (chapel - np.array([0, 0, 0])) * ease[:, None]  # (N,3)
    sway_amp = 0.3 * max(abs(w1[0]), abs(w3[0]))  # 30% of pin-spread
    sway = np.zeros_like(forward)
    sway[:, 0] = sway_amp * np.sin(2 * math.pi * ts)
    cam_pos = forward + sway  # (N,3), world coords

    # Camera always facing -Z (north): R = identity for camera-to-world
    # w2c = inv([R|t_cam_world]) with R = I, t_cam_world = cam_pos
    w2c = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    w2c[:, :3, 3] = -cam_pos  # since R = I, t_w2c = -t_cam_world

    fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    fy = fx
    K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float32)
    intr = np.tile(K, (num_frames, 1, 1))

    return w2c.astype(np.float32), intr, cam_pos


def main():
    pano = np.array(Image.open(PANO).convert("RGB"))
    first = equirect_to_perspective(pano, H, W, HFOV_DEG, yaw_deg=0.0, pitch_deg=0.0)
    Image.fromarray(first).save(OUT / "first_frame.png")
    print(f"saved {OUT / 'first_frame.png'}  {first.shape}")

    w2c, intr, cam_pos = build_trajectory(NUM_FRAMES, H, W, HFOV_DEG)
    np.savez(
        OUT / "trajectory.npz",
        w2c=w2c,
        intrinsics=intr,
        image_height=np.int32(H),
        image_width=np.int32(W),
    )
    print(f"saved {OUT / 'trajectory.npz'}  w2c {w2c.shape}  intr {intr.shape}")
    print(f"  cam_pos start={cam_pos[0]}  mid={cam_pos[len(cam_pos)//2]}  end={cam_pos[-1]}")

    captions = {
        "0": "A small stone chapel in a forest clearing, watercolor anime style, soft "
             "light filtering through trees, in the style of Witch Hat Atelier, painterly "
             "brushwork, warm dappled sunlight.",
        "81": "The camera glides forward toward the chapel doorway, revealing weathered "
              "stone steps and stained-glass windows, leaves drifting in the foreground, "
              "Witch Hat Atelier watercolor style.",
    }
    (OUT / "captions.json").write_text(json.dumps(captions, indent=2))
    print(f"saved {OUT / 'captions.json'}")


if __name__ == "__main__":
    main()
