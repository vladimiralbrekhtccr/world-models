# CLAUDE.md — handoff for the next agent

This is an **exploration folder** for world-model experiments. Each
experiment lives in its own subfolder (e.g. `3DGS/`). Read this file, then
the experiment-specific `README.md`.

## Execution environment — IMPORTANT
- **Do NOT use SLURM / sbatch / srun.** The user holds an interactive
  allocation. Just `ssh node007` and run commands there.
- **Use GPU index 0 only.** Always set `CUDA_VISIBLE_DEVICES=0` — other
  GPUs on node007 are in use by other workloads.
- `nvcc` and the CUDA toolkit are available on `node007`, not on the login
  node. Build any CUDA extensions from inside `ssh node007`.
- **Python envs use `uv`**, not conda. Each experiment carries its own
  `.venv/` built by its `env_setup.sh`.

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
  `ssh node007`.

## video → 3DGS with nerfstudio — READ BEFORE TOUCHING SPLATS

The canonical, working pipeline for a phone video → explorable 3D
gaussian splat. Do NOT reinvent any of these stages.

**Env** (already built, ~30 GB, on scratch — not home):
`/scratch/vladimir_albrekht/projects/world-models/MultiPano/.conda/nerfstudio`
- python 3.10 CPython, torch 2.1.2+cu118, gsplat 1.4.0, nerfstudio 1.1.5
- colmap 3.10 (conda-forge — 3.13 dropped `--SiftExtraction.use_gpu` which
  nerfstudio still passes), nvcc 11.8 + gcc 11 (cu118 nvcc rejects gcc>11),
  ffmpeg/ffprobe via `static-ffmpeg`
- `libcudart.so` is symlinked to torch's bundled cu11 runtime

**Pipeline** (`MultiPano/run_nerfstudio.sh` does steps 1-2):
1. `ns-process-data video --data <vid> --output-dir <proc> --num-frames-target 200
   --matching-method exhaustive` — ffmpeg extract + COLMAP SfM.
2. `ns-train splatfacto --data <proc> --max-num-iterations 30000` — ~4 min on H100.
3. **To get a PLY**: `MultiPano/export_via_nerfstudio.py` (runs nerfstudio's
   own `ExportGaussianSplat` with `pymeshlab` stubbed out — pymeshlab's
   import is broken in this env but is only needed for *mesh* export).
4. **To get a video**: `ns-render interpolate --load-config <cfg>
   --pose-source train --interpolation-steps N`.

### Mistakes made on 2026-05-18 — do NOT repeat
- **Do NOT hand-write a splatfacto→PLY exporter.** splatfacto stores SH
  rest coeffs as `(N,K,3)`; the INRIA/standard PLY wants `(N,3,K)` — needs
  a `.transpose(1,2)`. Also splat coords are in nerfstudio's *normalised*
  dataparser frame, not COLMAP world. Getting any of this wrong renders as
  noise needles. Just use `export_via_nerfstudio.py`.
- **Verify training FIRST with `ns-render` before assuming anything is
  broken.** On 2026-05-18 a bad custom PLY export was misdiagnosed as a
  training failure → wasted ~1.5 h GPU on COLMAP MVS + 25 min CPU on
  Poisson meshing as a "workaround" for a 1-line export bug.
- **Don't pivot to mesh/MVS to dodge a splat bug.** Mesh is a separate
  deliverable, not a fix for a broken splat export.
- ns-export's `gaussian-splat` subcommand can't even be reached because
  `nerfstudio.scripts.exporter` top-level imports `pymeshlab` → stub it
  (see `export_via_nerfstudio.py`).

## kitan-a.com deploy
- foggen serves `kitan-a.com` via Caddy. Static routes live under
  `/var/www/<name>/` with a `handle_path /<name>/*` block in
  `/etc/caddy/Caddyfile` (+ a `redir /<name> /<name>/ 308`).
- `/var/www/3dgs/` is already wired up — scp files there, served at
  `https://kitan-a.com/3dgs/...`. No 100 MB limit (unlike GH Pages).
