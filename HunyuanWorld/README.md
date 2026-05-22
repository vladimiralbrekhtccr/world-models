# HunyuanWorld 2.0 — what works on one H100

Single-GPU exploration of Tencent's [HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)
on node001 GPU 6. **Live viewer:** <https://kitan-a.com/3dgs/hyworld/>
(switch between 3 scenes in the dropdown — Statue Face, Small Room, Tree+Building).

The heavy artifacts (upstream clone, conda env, HF cache, sweep outputs,
ksplats) live at `/scratch/vladimir_albrekht/projects/hunyuanworld/`
— this folder under `world-models/` is the version-controlled record:
README + the three install/run scripts. Run them from the scratch sibling
(or copy them there) — paths are absolute to scratch.

## What's installed

`/scratch/.../hyworld2` — python 3.11.15, torch 2.7.1+cu128, gsplat 1.5.3
(their `gsplat_maskgaussian` fork), flash-attn 2.8.3, diffusers 0.36.0,
transformers 5.2.0. Built by `setup_hyworld.sh` in one shot.

Things that bit us during install:
- `cupy==13.6.0` (source) wouldn't build → swap to `cupy-cuda12x==13.6.0`
  (prebuilt wheel) in `HY-World-2.0/requirements.txt`.
- `gsplat_maskgaussian` needs **GLM** headers → `conda install -c
  conda-forge glm` + `export CPATH=$PREFIX/include:$CPATH`.
- CU128 `nvcc` rejects gcc > 11 → pin `gcc_linux-64=11`.

Cache redirection (set in every `run_*.sh`):
```
HF_HOME       → ./.cache/hf
PIP_CACHE_DIR → ./.cache/pip
TORCH_HOME    → ./.cache/torch
XDG_CACHE_HOME→ ./.cache/xdg
TMPDIR        → ./.tmp
CONDA_PKGS_DIRS→ ./.conda/pkgs
```

## What works on one GPU

### WorldMirror 2.0 — multi-view → 3DGS (feed-forward, no optimization)

```bash
bash run_worldmirror.sh examples/worldrecon/realistic/Desk
# or sweep all examples:
bash sweep_worldmirror.sh
```

The pipeline auto-downloads `tencent/HY-World-2.0` (~3 files, ~5GB) to
`HF_HOME` on first run. Model takes **68s** to load, then per-scene
inference is **~1s** (2 imgs @ 952×714, ~800K gaussians).

**VRAM: 2.35GB allocated** — fits anywhere.

Outputs per scene (in a timestamped subdir):
- `gaussians.ply` — the 3DGS (post-voxel-prune)
- `points.ply` — colored point cloud
- `depth/depth_NNNN.{png,npy}`
- `normal/normal_NNNN.png`
- `camera_params.json`
- `pipeline_timing.json`

Deployed to kitan-a.com — see the **Deployment** section below.

## What does NOT fit on one GPU (skipped)

- **HY-Pano 2.0** — ~80B params text/image → panorama. Way over 80GB
  even at fp16. Document only.
- **WorldGen pipeline** (`hyworld2/worldgen`) — full panorama → navigable
  3DGS world. The 5-stage pipeline requires:
  - a separate **vLLM server** running **Qwen3-VL-8B-Instruct**
    (`--tensor-parallel-size 8`) for VLM-guided trajectory planning
    (stages 1+2),
  - **8-GPU FSDP** for WorldStereo 2.0 (~17B) keyframe generation
    (stage 3),
  - multi-GPU for `gen_gs_data.py` + `world_gs_trainer.py` (stages 4+5).
  Per the project rule "if 1 GPU isn't enough, just stop" — **skipped**.

  We could probably get `video_gen.py` (WorldStereo) to load on one
  H100 (~34GB fp16), but stages 1+2 are a hard blocker without a
  Qwen3-VL deployment. If you want to come back to this, you'd need:
  (a) stand up vLLM on a separate GPU pool, (b) modify `video_gen.py`
  to drop `--fsdp` and run single-GPU, (c) lower max_steps in stage 5
  per the README's "x1 GPU: max_steps 8000" guidance.

## Files

```
hunyuanworld/
├── README.md               ← this file
├── setup_hyworld.sh        ← env build (idempotent)
├── run_worldmirror.sh      ← one scene
├── sweep_worldmirror.sh    ← all realistic examples, model loaded once
├── HY-World-2.0/           ← upstream clone (requirements.txt patched)
├── .conda/hyworld2/        ← conda env
├── .cache/                 ← all redirected caches
├── .tmp/
└── output/
    ├── worldmirror/        ← single-scene runs
    └── worldmirror_sweep/  ← sweep results
```

## Sweep results — 14/14 succeeded on GPU 6

Model loaded once (`[Memory] allocated=2.35GB`), then 14 scenes pushed
through interactive stdin. Total disk: 2.6GB of PLYs.

| scene | imgs | resolution | inference | total wall | gaussians | ply |
|---|---|---|---|---|---|---|
| Desk           |  2 | 714×952 | 0.77s | 7.9s  | 786K  | 52M  |
| Dining_Table   |  4 | 700×756 | 0.40s | 10.2s | 811K  | 53M  |
| Park           |  3 | 546×952 | 0.37s | 7.9s  | 784K  | 51M  |
| Workspace      |  4 | 434×756 | 0.34s | 6.7s  | 473K  | 31M  |
| Room_Cat       |  8 | 378×504 | 0.33s | 10.7s | 1.13M | 74M  |
| Valley         | 11 | 378×672 | 0.53s | 15.0s | 1.29M | 85M  |
| Building       | 32 | 504×504 | 1.47s | 42.4s | 3.30M | 215M |
| Statue_Face    | 32 | 504×504 | 1.31s | 39.7s | 2.28M | 148M |
| Small_Room     | 32 | 630×952 | 5.05s | 88.3s | 4.90M | 318M |
| Messy_Room     | 32 | 630×952 | 5.03s | 85.0s | 4.91M | 319M |
| Park_Stone     | 32 | 504×504 | 1.31s | 30.8s | 1.21M | 79M  |
| Tree_Building  | 32 | 504×504 | 1.35s | 44.3s | 3.89M | 253M |
| Landmark       | 32 | 504×504 | 1.36s | 35.7s | 2.38M | 155M |
| Archway_Tunnel |  2 | 434×756 | 0.25s | 3.9s  | 325K  | 22M  |

Takeaways:
- **Inference is fast** (sub-second up to 32 imgs at 504², ~5s at 952×630×32).
  The expensive part is the post-process mask-filter + voxel-prune
  (~4s) and PLY serialization (~3-30s scaled with gaussian count).
- **No OOM at any point** — VRAM stayed near 2.35GB throughout. The
  model is tiny; what scales is the *number of output gaussians*
  (= num_imgs × H × W after their down-stride).
- 32-image scenes at 952×630 are the practical ceiling for one PLY save
  on 80GB scratch budget — `Small_Room` is 18.4M gaussians pre-prune.

## Deployment — kitan-a.com/3dgs/hyworld/

Three sweep scenes are live in a Spark viewer with a dropdown switcher:
**Statue_Face** (2.3M gs, 53M ksplat), **Small_Room** (4.9M gs, 113M),
**Tree_Building** (3.9M gs, 90M).

Pipeline PLY → deployed scene:

1. **ksplat conversion** — `MultiPano/ksplat/convert.mjs` with node20:
   ```
   node convert.mjs <gaussians.ply> <out.ksplat> 1 0.01 0
   ```
   - compression `1` (16-bit, visually lossless, ~65% smaller).
   - `shDeg 0` — WorldMirror PLYs carry only `f_dc_*` (no `f_rest`),
     i.e. spherical-harmonics degree 0, same as Lyra.
2. **coord convention** — WorldMirror outputs are **OpenCV (Y-down)**.
   In the Spark viewer rotate the SplatMesh `rotation.set(Math.PI,0,0)`
   (X = π) — same flip as Lyra PLYs. (Splatfacto PLYs differ: +Z-up,
   `-X/2`.)
3. **upload** — `scp` the ksplats + `index.html` to
   `foggen:/var/www/3dgs/hyworld/`.
4. **Caddy route** — `/var/www/3dgs/` is already wired with a generic
   `handle_path /3dgs/*` block. A new sub-path only needs a redirect so
   the bare URL doesn't 404:
   ```
   redir /3dgs/hyworld /3dgs/hyworld/ 308
   ```
   added to `/etc/caddy/Caddyfile`, then `sudo systemctl reload caddy`.

Viewer (`output/ksplat/index.html`): Spark `SplatMesh`, dual control —
orbit (default, auto-spin + zoom clamps) and fly (WASD move, QE roll,
RF up/down, drag-look). **Fly-mode roll fix:** the camera quaternion is
rebuilt every frame from `yaw`/`pitch`, so roll must be tracked as its
own state variable and baked into the `Euler(pitch, yaw, roll, 'YXZ')`
— a one-shot `camera.rotateZ()` gets wiped on the next frame.

## Related files in this repo

- `HUNYUANWORLD_PIPELINE.md` — the full 5-stage HY-World pipeline
  (HY-Pano → WorldNav → WorldStereo → WorldMirror → gsplat trainer),
  plus an infographic image-generation brief.
- `CLAUDE.md` section F — the condensed WorldMirror recipe.
