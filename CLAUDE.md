# CLAUDE.md — handoff for the next agent

This is an **exploration folder** for world-model experiments. Each
experiment lives in its own subfolder (e.g. `3DGS/`). Read this file, then
the experiment-specific `README.md`.

## Execution environment — IMPORTANT
- **Do NOT use SLURM / sbatch / srun.** The user holds an interactive
  allocation. Just `ssh node001` and run commands there.
- **Use GPU index 6 only** (as of 2026-05-18). Always set
  `CUDA_VISIBLE_DEVICES=6` — other GPUs on node001 are in use by other
  workloads. (The MultiPano `run_*.sh` scripts still hardcode
  `CUDA_VISIBLE_DEVICES=0` from earlier — update them to 6 before reuse.)
- `nvcc` / CUDA toolkit live inside the per-experiment conda envs, not on
  the login node. Build CUDA extensions from inside `ssh node001`.
- **Python envs:** `3DGS/` uses a `uv` `.venv`; `MultiPano/` uses conda
  envs on scratch (nerfstudio, lyra2) — see the nerfstudio section below.

## File-creation rule — IMPORTANT
**Do not sprawl files.** Keep each experiment lean: one script per path,
one README, plus outputs in `out/`. Do not create logs, notes, plan docs,
scratch files, or "v2"/"old"/"backup" copies. Edit existing files instead
of forking them. If you need to capture intermediate state, use the
conversation, not the filesystem.

## Current experiments

- **`3DGS/`** — one 360° panorama → 3D Gaussian Splatting scene.
  See `3DGS/README.md` for the full design (depth-only fast path,
  DreamScene360 full path) and `3DGS/env_setup.sh` to build the uv venv.

## Style and behaviour notes
- Terse responses; no trailing summaries.
- Propose first, wait for thumbs-up, then implement.
- **Pushing to the user's own repos is pre-authorized** — including
  `vladimiralbrekhtccr/world-models` and `vladimiralbrekhtccr/360-panorama-viewer`.
  Just push additive, sensible changes without asking each time. Still
  confirm before destructive ops (force-push, history rewrites, deleting
  branches, removing existing files).

## Broader context
- Panorama viewer (already deployed):
  <https://github.com/vladimiralbrekhtccr/360-panorama-viewer>
- The user is on macOS, reaches this cluster via `ssh foggen` → `ssh US` →
  `ssh node001`.
- Project direction: build a World-Labs / Marble-style "image/prompt →
  explorable 3D world". See `READING.md` for the paper list.

# =====================================================================
# 3DGS PLAYBOOK — everything below was validated 2026-05-18/19. Follow it.
# =====================================================================

## Envs (all on scratch, NOT home — home was 99% full)
- **nerfstudio**: `MultiPano/.conda/nerfstudio` — python 3.10 CPython,
  torch 2.1.2+cu118, gsplat 1.4.0, nerfstudio 1.1.5, colmap 3.10
  (conda-forge; 3.13/4.x dropped `--SiftExtraction.use_gpu` which
  nerfstudio still passes), nvcc 11.8 + gcc 11 (cu118 nvcc rejects
  gcc>11), ffmpeg/ffprobe via `static-ffmpeg`. `libcudart.so` symlinked
  to torch's bundled cu11 runtime.
- **lyra2**: `/scratch/vladimir_albrekht/projects/lyra/.conda/lyra2` —
  Lyra 2.0 + gsplat 1.5.2 + colmap (pycolmap) + open3d. Used for Lyra
  inference AND for the gsplat-optimization / debug-render steps.
- **node**: `~/miniconda3/envs/node20/bin/node` — for the ksplat tool.

## A. phone video → 3DGS splat  (the main, highest-quality path)
`MultiPano/run_nerfstudio_big.sh <video> <proj-dir> <exp-name> [gpu=6] [n-frames=400]`
does all of: extract → COLMAP → splatfacto-big 30K → PLY → ksplat →
flythrough. Stages:
1. `ns-process-data video … --matching-method exhaustive` — ffmpeg
   extract N frames + COLMAP SfM. COLMAP incremental mapping is the slow
   CPU stage (~10 min at 720p; ~1 h at 4K — 4K images = ~14k SIFT
   features each → heavy bundle adjustment).
2. `ns-train splatfacto-big` — 30K iters, ~18 min on H100. `splatfacto-big`
   (not plain `splatfacto`) = ~3-4× more gaussians = sharper. Plain
   splatfacto for a fast/small result.
3. PLY export: `MultiPano/export_via_nerfstudio.py` — runs nerfstudio's
   own `ExportGaussianSplat` with `pymeshlab` stubbed (pymeshlab's import
   is broken in-env but only needed for *mesh* export).
4. ksplat: see section C.
5. flythrough: `ns-render interpolate --pose-source train
   --interpolation-steps N`.
- ~205 extracted frames → COLMAP typically registers ~160 (blurry frames
  dropped). More frames = better coverage; mpdecimate drops dup/blur.

## B. recording recipe (tell the user this before they film)
- **4K30**, 1× lens ONLY (never switch zoom — different lens = different
  intrinsics → COLMAP breaks). Normal mode, **HDR off**, **stabilization/
  "Ultra Steady" (EIS) off** (EIS warps frames non-rigidly). AF+AE
  **locked** on the subject. Slow, smooth motion.
- **Single object** (e.g. a plant): orbit *around* it at steady radius —
  best case, easy to reconstruct.
- **Room/scene**: walk through slowly; to extend later, record a 2nd
  video that **visually overlaps** the first, then COLMAP+train both
  together (joint SfM — you cannot append to a finished splat).

## C. PLY → .ksplat  (deploy format)
`MultiPano/ksplat/convert.mjs` (run with node20):
`node convert.mjs <in.ply> <out.ksplat> <compression> <alphaThresh> <shDeg>`
- compression **1** = 16-bit, visually lossless, ~70% smaller, fast parse
  — USE THIS. compression 0 (true-lossless) does NOT render in Spark.
- shDeg: splatfacto PLYs = 2 (ksplat caps SH at 2); Lyra PLYs = 0.
- ksplat modules are prebuilt files in `MultiPano/ksplat/` (gs3d.module.js
  patched so `import 'three'` → local file) — do NOT `npm install`.

## D. Lyra 2.0 → splat  (the WORKING recipe — validated 2026-05-19)
Lyra = single image → diffusion-generated 3D scene, no COLMAP.
**Key lesson: Lyra's own Step-2 feed-forward recon is a soft transparent
cloud — do NOT ship it. Use Lyra only to GENERATE the video, then run
real gsplat optimization on that video.**
1. Build trajectory: `MultiPano/build_lyra_inputs_arc.py` — camera ORBITS
   AROUND the subject (translating circle, not in-place spin — in-place
   spin gives zero parallax → no usable depth), **481 frames** (Lyra max).
2. `MultiPano/run_lyra_arc.sh` — Lyra Step 1 (video diffusion) + Step 2
   (`vipe_da3_gs_recon`, kept only for its VIPE camera poses + an init PLY).
3. `MultiPano/train_gsplat_vipe.py` — gsplat photometric+SSIM optimization
   on Lyra's video using the VIPE poses, 30K iters, DefaultStrategy
   densification. THIS is what makes it crisp.
4. ksplat (section C, shDeg 0) → deploy.
Result: 1.5M optimized gaussians, a real navigable chapel. The chain is:
**diffusion imagines the scene → optimization makes it a clean splat.**

## E. kitan-a.com deploy
- foggen serves `kitan-a.com` via Caddy. Static routes: `/var/www/<name>/`
  + a `handle_path /<name>/*` block in `/etc/caddy/Caddyfile` + a
  `redir /<name> /<name>/ 308` (bare path 404s without the redir).
- `/var/www/3dgs/` is wired up — `scp` files there → served at
  `https://kitan-a.com/3dgs/...`. No size limit (unlike GH Pages).
- Viewer pages use **Spark** (World Labs' 3DGS renderer, `sparkjs.dev`)
  or the mkkellogg viewer. The plant pages have a dual orbit/fly control
  (orbit = easy default w/ auto-spin + zoom clamps; fly = WASD + Q/E roll
  + R/F up-down + drag-look). Splatfacto PLYs are +Z-up → rotate the
  SplatMesh `-X/2`; Lyra PLYs are OpenCV Y-down → rotate `X=π`.
- Live demos: `/3dgs/plant/` (720p), `/3dgs/plant-4k/`, `/3dgs/lyra/`,
  `/3dgs/` (room), `/3dgs/colmap/` (COLMAP dense MVS point cloud).

## Mistakes made — do NOT repeat
- **Don't hand-write a splatfacto→PLY exporter.** SH rest coeffs need a
  `.transpose(1,2)` for INRIA PLY layout; coords are in the normalised
  dataparser frame. Use `export_via_nerfstudio.py`.
- **Verify training with `ns-render` BEFORE assuming it's broken.** A bad
  custom PLY export was misdiagnosed as training failure → wasted ~1.5 h
  GPU on COLMAP MVS + Poisson meshing for a 1-line export bug.
- **Don't pivot to mesh/MVS to dodge a splat bug.** Separate deliverable.
- **Don't ship Lyra's feed-forward recon directly** — optimize the video.
- **Detach long jobs properly**: `nohup setsid bash <abs-path> > <abs-log>
  2>&1 < /dev/null &`. ssh-session teardown can SIGHUP a COLMAP grandchild
  otherwise (a 4K run died mid-finalize this way; COLMAP output survived,
  resumed with `--skip-colmap`). And the `>log` redirect resolves in the
  ssh cwd ($HOME) — use absolute paths for both script and log.
- **COLMAP mapping is CPU-bound and slow** (no GPU help). 4K → ~1 h.
  GLOMAP graduated into COLMAP 4.0 as `colmap global_mapper` (10-50×
  faster, global SfM) — but COLMAP 4.x's CLI is incompatible with
  nerfstudio 1.1.5, so it'd need a separate env + manual wiring.
