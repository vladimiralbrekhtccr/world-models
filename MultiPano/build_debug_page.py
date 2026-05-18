"""Build a side-by-side debug page so we can SEE what went wrong.

For 8 evenly-spaced training frames:
  - col 1: input image (as ffmpeg gave us)
  - col 2: COLMAP MVS depth map (geometric) coloured
  - col 3: gsplat-rendered view from the splatfacto PLY at this exact pose

If col 1 is blurry → recording problem.
If col 2 has huge holes / wrong shapes → matching/SfM problem.
If col 3 is junk → splatfacto training problem.
If all three are sharp but the final mesh is junk → meshing problem.
"""
import json
import math
import struct
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gsplat

SH_C0 = 0.28209479177387814


def read_colmap_depth(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        # header is ascii "W&H&C&", then a single newline-style sep — actually
        # it's just "W&H&C&" with no terminator. Parse char-by-char.
        header = b""
        amps = 0
        while amps < 3:
            c = f.read(1)
            header += c
            if c == b"&":
                amps += 1
        w, h, c = header.decode().strip("&").split("&")
        w, h, c = int(w), int(h), int(c)
        data = np.frombuffer(f.read(), dtype=np.float32).reshape((h, w, c))
    return data[..., 0] if c == 1 else data


def colorize_depth(d: np.ndarray) -> np.ndarray:
    valid = d > 0
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(d[valid], [2, 98])
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    n[~valid] = 0
    cmap = matplotlib.colormaps["turbo"]
    rgba = cmap(n)
    rgba[~valid] = [0, 0, 0, 1]
    return (rgba[..., :3] * 255).astype(np.uint8)


def render_splat(ply_path: Path, w2c: np.ndarray, K: np.ndarray,
                 W: int, H: int, device="cuda") -> np.ndarray:
    """Render a single view from the splatfacto PLY using gsplat."""
    v = PlyData.read(str(ply_path))["vertex"]
    pos = torch.tensor(np.stack([v["x"], v["y"], v["z"]], -1), dtype=torch.float32, device=device)
    quats = torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], -1), dtype=torch.float32, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.tensor(np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1)), dtype=torch.float32, device=device)
    opac = torch.tensor(1.0 / (1.0 + np.exp(-v["opacity"])), dtype=torch.float32, device=device)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1).astype(np.float32)
    rgb = torch.tensor(np.clip(fdc * SH_C0 + 0.5, 0, 1), dtype=torch.float32, device=device)

    vm = torch.from_numpy(w2c.astype(np.float32)).to(device).unsqueeze(0)
    Kt = torch.from_numpy(K.astype(np.float32)).to(device).unsqueeze(0)
    img, _, _ = gsplat.rasterization(
        means=pos, quats=quats, scales=scales, opacities=opac, colors=rgb,
        viewmats=vm, Ks=Kt, width=W, height=H,
        sh_degree=None, render_mode="RGB",
    )
    return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True,
                    help="ns_room2/processed (has images/, colmap/, dense/)")
    ap.add_argument("--ply", type=Path, required=True, help="splatfacto PLY")
    ap.add_argument("--dataparser_transform", type=Path, default=None,
                    help="nerfstudio dataparser_transforms.json (splat lives in normalised frame)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    args = ap.parse_args()

    # Load dataparser transform (rotation+translation) and scale.
    dp_T = None; dp_s = 1.0
    if args.dataparser_transform and args.dataparser_transform.exists():
        dp = json.load(open(args.dataparser_transform))
        dp_T = np.array(dp["transform"], dtype=np.float64)  # (3,4)
        dp_s = float(dp["scale"])
        print(f"dataparser: scale={dp_s:.4f}, R={dp_T[:3,:3].round(3).tolist()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "img").mkdir(exist_ok=True)
    (args.out_dir / "depth").mkdir(exist_ok=True)
    (args.out_dir / "splat").mkdir(exist_ok=True)

    transforms = json.load(open(args.processed / "transforms.json"))
    frames = transforms["frames"]
    # transforms.json camera params (one shared camera per ns-process-data)
    fl_x = transforms.get("fl_x"); fl_y = transforms.get("fl_y", fl_x)
    cx = transforms.get("cx"); cy = transforms.get("cy")
    W = int(transforms.get("w")); H = int(transforms.get("h"))
    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)

    n = len(frames)
    sample_idx = np.linspace(0, n - 1, args.n_samples).astype(int)
    rows = []
    for s, i in enumerate(sample_idx):
        f = frames[int(i)]
        name = Path(f["file_path"]).name  # "frame_00037.png"
        # column 1: input image
        src = args.processed / "images" / name
        if not src.exists():
            print(f"skip {name}: missing image"); continue
        Image.open(src).convert("RGB").resize((W // 2, H // 2)).save(args.out_dir / "img" / f"{s:02d}.jpg", "JPEG", quality=85)

        # column 2: MVS depth map
        dpath = args.processed / "dense" / "stereo" / "depth_maps" / f"{name}.geometric.bin"
        if dpath.exists():
            d = read_colmap_depth(dpath)
            Image.fromarray(colorize_depth(d)).resize((W // 2, H // 2)).save(args.out_dir / "depth" / f"{s:02d}.jpg", "JPEG", quality=85)
            depth_status = "ok"
        else:
            Image.new("RGB", (W // 2, H // 2), color=(40, 40, 40)).save(args.out_dir / "depth" / f"{s:02d}.jpg", "JPEG", quality=85)
            depth_status = "MISSING"

        # column 3: splatfacto render at this exact pose.
        # transforms.json stores c2w in nerfstudio OpenGL frame (+X right,
        # +Y up, -Z forward). gsplat / our splat PLY use OpenCV frame
        # (+X right, +Y down, +Z forward). And the gaussians live in the
        # *normalised* coord system after the dataparser transform, so apply
        # that to the camera position too.
        c2w = np.array(f["transform_matrix"], dtype=np.float64)
        if dp_T is not None:
            # Apply dataparser transform to the camera-to-world pose.
            # T = [[R | t]; 0 0 0 1]; world_new = R @ world_old + t; then scale.
            T4 = np.eye(4, dtype=np.float64); T4[:3, :4] = dp_T
            c2w = T4 @ c2w
            c2w[:3, 3] *= dp_s
        # nerfstudio camera convention → OpenCV (flip Y and Z basis vectors)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        c2w_opencv = c2w @ flip
        w2c = np.linalg.inv(c2w_opencv)
        try:
            render = render_splat(args.ply, w2c, K, W, H)
            Image.fromarray(render).resize((W // 2, H // 2)).save(args.out_dir / "splat" / f"{s:02d}.jpg", "JPEG", quality=85)
            splat_status = "ok"
        except Exception as e:
            print(f"  splat render {s} failed: {e}")
            Image.new("RGB", (W // 2, H // 2), color=(60, 0, 0)).save(args.out_dir / "splat" / f"{s:02d}.jpg", "JPEG", quality=85)
            splat_status = "FAILED"

        rows.append({"i": int(i), "name": name, "depth": depth_status, "splat": splat_status})
        print(f"  [{s+1}/{args.n_samples}] frame {i:3d} {name}  depth={depth_status}  splat={splat_status}")

    html = ['<!doctype html><html><head><meta charset="utf-8">',
            '<title>3DGS pipeline debug</title>',
            '<style>',
            'body{font:13px/1.4 ui-monospace,monospace;background:#0a0a0a;color:#ddd;margin:0;padding:18px;}',
            'h1{color:#fff;margin:0 0 8px}',
            'p{color:#aaa;max-width:900px}',
            'table{border-collapse:collapse;margin-top:14px;width:100%;}',
            'th,td{padding:6px 8px;vertical-align:top;border-bottom:1px solid #222;text-align:left}',
            'th{background:#181818;font-weight:600;color:#fff}',
            'img{display:block;max-width:100%;height:auto}',
            'code{background:#1a1a1a;padding:1px 6px;border-radius:3px;color:#9d9}',
            '</style></head><body>',
            f'<h1>3DGS pipeline debug · {len(rows)} samples</h1>',
            '<p>Three views per row: <b>input image</b> (what ffmpeg extracted) · '
            '<b>COLMAP MVS depth</b> (geometric pass, turbo colormap) · '
            '<b>splatfacto rendered view</b> (PLY rasterised from the same camera pose, no checkpoint required).<br>'
            'If input is blurry → recording problem. '
            'If depth has huge holes → matching/SfM problem. '
            'If splat render is wildly different from input → training problem.</p>',
            '<table>',
            '<tr><th style="width:60px">frame</th><th>input</th><th>MVS depth</th><th>splat render</th></tr>']
    for s, r in enumerate(rows):
        html.append(f'<tr><td><code>{r["name"]}</code><br><br>'
                    f'depth: {r["depth"]}<br>splat: {r["splat"]}</td>'
                    f'<td><img src="img/{s:02d}.jpg"></td>'
                    f'<td><img src="depth/{s:02d}.jpg"></td>'
                    f'<td><img src="splat/{s:02d}.jpg"></td></tr>')
    html.append('</table></body></html>')
    (args.out_dir / "index.html").write_text("\n".join(html))
    print(f"\nwrote {args.out_dir}/index.html  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
