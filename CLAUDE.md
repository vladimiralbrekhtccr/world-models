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
