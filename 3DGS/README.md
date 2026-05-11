# World Models — Panorama → 3D Gaussian Splatting

Take a single 360° equirectangular panorama (`panorama.png`) and produce an
explorable 3D Gaussian Splatting scene (`.ply` of Gaussians) that can be
opened in any 3DGS viewer or shipped to a browser.

Target hardware: **node007** (1× H100 80GB) on the local HPC.

---

## The honest catch

A single image has **no parallax**, so depth is not measurable, only
inferred. Everything we produce is partly hallucinated by a neural prior:

- regions visible in the panorama get plausible depth from a monocular
  depth model
- regions occluded from the panorama get inpainted by a 2D diffusion model
- novel views close to the original camera look great
- novel views far from the original camera degrade sharply

This is fundamental — not a bug in any specific tool. If we ever wanted
true geometric 3D we'd need multiple panoramas from different positions.

---

## Two paths

We support a **fast path** (depth-only, no diffusion) and a **full path**
(DreamScene360-style, with diffusion refinement). Iterate fast on Path A,
graduate to Path B once we like the output.

### Path A — depth-only (fast, ~3–10 min on one H100)

```
panorama.png  ─►  DepthAnything-V2  ─►  depth map
                                          │
                                          ▼
                              unproject by spherical coords
                                          │
                                          ▼
                          isotropic Gaussian per pixel
                                          │
                                          ▼
                                    out/fast.ply
```

1. Run DepthAnything-V2-Large on `panorama.png` (treats it as a normal
   image — imperfect for equirectangular but functional as a baseline).
2. For each pixel `(u, v)` at depth `d`, compute spherical coordinates
   `lon = 2π·u/W − π`, `lat = π/2 − π·v/H`, then
   `(x, y, z) = d · (cos·sin, sin, cos·cos)`.
3. Emit one small isotropic Gaussian per pixel (color = pixel RGB,
   opacity ≈ 0.9, scale ≈ 5 cm).
4. Write `out/fast.ply` in the canonical 3DGS PLY schema.

Run: `python run_depth_fast.py`

**What it looks like:** good from near the original camera, with visible
"depth skirts" at object silhouettes (a known artifact of depth-only
methods — there's no information about what's behind a foreground edge).

### Path B — DreamScene360 (full, ~1–3 h on one H100)

```
panorama  ─►  20× perspective slices  ─►  per-view depth
                                            │
                                            ▼
                           initialize 3DGS from point cloud
                                            │
                                            ▼
                photometric loss on 20 views + SDS loss from
                a 2D diffusion prior to fill occluded regions
                                            │
                                            ▼
                                    out/scene.ply
```

1. Slice the equirectangular image into ~20 perspective views.
2. Estimate depth per view; back-project to a unified point cloud.
3. Initialize 3D Gaussians.
4. Optimize with two losses:
   - photometric reconstruction on the 20 ground-truth views,
   - **Score Distillation Sampling** (SDS) with Stable Diffusion 2.1 as
     prior, applied to randomly sampled novel views to fill in unseen
     regions.
5. Export `out/scene.ply` + an MP4 fly-through render.

Run: `bash run_dreamscene360.sh`

Reference: *DreamScene360*, Zhou et al., ECCV 2024.
Repo URL is hard-coded in the script; verify before running (GitHub
project name may have moved).

---

## Folder layout (once populated)

```
world-models/3DGS/
├── README.md                  ← this file
├── panorama.png               ← input (2:1 equirectangular, 2048×1024)
├── env_setup.sh               ← one-shot uv venv creation
├── run_depth_fast.py          ← Path A
├── run_dreamscene360.sh       ← Path B
├── .venv/                     ← uv venv, gitignored
├── out/                       ← created by runs; .ply / .splat / .mp4
└── third_party/               ← created by Path B; cloned repos
```

---

## First run (recommended order)

```bash
ssh foggen
ssh US
ssh node007                          # user holds an interactive allocation
cd /scratch/vladimir_albrekht/projects/world-models/3DGS

# 1. One-time: build the uv venv in 3DGS/.venv
bash env_setup.sh

# 2. Activate and run Path A on GPU 0
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python run_depth_fast.py

# 3. Inspect output
ls -la out/
#   fast.ply   ← drop into any 3DGS viewer
```

If Path A looks promising, run `bash run_dreamscene360.sh` for Path B.

---

## Viewing the result

- **Desktop:** open `out/*.ply` in
  [SuperSplat](https://playcanvas.com/super-splat),
  [SplatViewer](https://github.com/antimatter15/splat), or any modern
  3DGS viewer.
- **Browser:** convert `.ply` → `.splat` and serve from the same GitHub
  Pages repo we already deployed the panorama viewer to
  (`vladimiralbrekhtccr/360-panorama-viewer`). The browser viewer is
  ~20 KB of JS and runs everywhere.

---

## Known unknowns / pitfalls

- **DreamScene360 repo URL** — verify in `run_dreamscene360.sh` before
  running. The original was at `ShijieZhou-UCLA/DreamScene360`; if moved,
  search GitHub.
- **Diffusion checkpoint license** — DreamScene360 uses SD 2.1. Check the
  license if we ever intend to publish generated content.
- **Equirectangular vs perspective depth** — DepthAnything-V2 is trained
  on perspective images. Running it directly on the equirectangular
  panorama (Path A) underperforms running it per-cube-face and stitching.
  TODO in `run_depth_fast.py` if we need higher quality.
- **VRAM** — H100 80GB has plenty of headroom for the default
  DreamScene360 config. Watch usage if we bump resolution or view count.
- **Inverse vs metric depth** — DepthAnything's pipeline returns relative
  inverse depth, not metric. Path A rescales it to an arbitrary
  [0.5 m, 30 m] range; results look fine, but absolute scale is
  meaningless. Doesn't matter for visual splat rendering, would matter for
  any geometric measurement.

---

## Roadmap beyond this folder

1. ✅ Browser panorama viewer (deployed to GH Pages).
2. ▶ Path A — depth-only Gaussian Splatting from one panorama.
3. ▷ Path B — DreamScene360 full pipeline.
4. ▷ Browser 3DGS viewer on the same GH Pages site (drop-in
   `gsplat.js` or `antimatter15/splat`).
5. ▷ Multi-panorama capture + true multi-view 3DGS (real geometry, no
   hallucination) — requires additional panorama renders from different
   spatial positions in the same scene.
