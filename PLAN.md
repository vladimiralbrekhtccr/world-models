# world-models — project plan

The **image/prompt → explorable 3D world** project (World-Labs / Marble
style). This file *is* the tracker — edit it and `git push`; GitHub
renders the tables. No Google Sheet, no API key, no sync script.

**Status:** ✅ done · 🔄 in progress · ⬜ todo · ⛔ blocked
**Target** dates are suggestions anchored to 2026-05-22 — edit freely.
Outward-facing intro: <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/project/>

---

## Roadmap

### Phase 1 — 3DGS foundations ✅

| Task | Status | Target | Notes |
|---|---|---|---|
| Single panorama → 3DGS splat + mesh (Path A) | ✅ | done | live on `360-panorama-viewer` Pages |
| splatfacto-big video → 3DGS pipeline | ✅ | done | `MultiPano/run_nerfstudio_big.sh`; plant 720p + 4K |
| PLY → ksplat compression | ✅ | done | `ksplat/convert.mjs`, compression 1 |
| kitan-a.com Caddy deploy + Spark viewer | ✅ | done | `/3dgs/plant/`, `/3dgs/plant-4k/` |
| READING.md — explorable-world paper list | ✅ | done | 8 papers |

### Phase 2 — MultiPano: multi-panorama → real 3DGS ✅

| Task | Status | Target | Notes |
|---|---|---|---|
| 3 overlapping panoramas (witch_hat_atelier) | ✅ | done | Codex `image_gen` — real equirect |
| Camera pin placement → poses.json | ✅ | done | `place-pins.html` |
| Depth-fusion baseline (`recon_fusion.py`) | ✅ | done | 1.18M-gaussian PLY, `/3dgs-multipano/` |
| Real multi-view 3DGS (`recon_3dgs.py`, gsplat) | ✅ | done | 590k gaussians, `/3dgs-multipano-mv/` |

### Phase 3 — Lyra 2.0 (diffusion → 3D) ✅

| Task | Status | Target | Notes |
|---|---|---|---|
| Lyra 2.0 install + env on scratch | ✅ | done | `lyra2` conda env |
| Arc-orbit trajectory recipe (481 frames) | ✅ | done | `build_lyra_inputs_arc.py` |
| Chapel: diffusion video → gsplat optimization | ✅ | done | 1.5M gaussians, `/3dgs/lyra/` |

### Phase 4 — HunyuanWorld 2.0 tests ✅

| Task | Status | Target | Notes |
|---|---|---|---|
| Build `hyworld2` env on scratch (1 GPU) | ✅ | done | torch 2.7.1+cu128, gsplat 1.5.3, flash-attn 2.8.3 |
| WorldMirror 2.0 single-scene test | ✅ | done | Desk: 2 imgs → 800K gs in ~1s, 2.35GB VRAM |
| WorldMirror sweep — 14 realistic examples | ✅ | done | 14/14 ok, no OOM |
| Deploy 3 scenes to kitan-a.com | ✅ | done | `/3dgs/hyworld/` dropdown switcher |
| Document pipeline + deployment | ✅ | done | `HUNYUANWORLD_PIPELINE.md`, `HunyuanWorld/README.md` |

### Phase 5 — splat → mesh ✅/🔄

| Task | Status | Target | Notes |
|---|---|---|---|
| garden.splat → Poisson mesh | ✅ | done | filters + Poisson depth 11 + largest-component |
| `garden-mesh-viewer` standalone repo | ✅ | done | github.com/vladimiralbrekhtccr/garden-mesh-viewer |
| Confirm mesh visual quality (v2) | 🔄 | 2026-05-23 | pending review on `/3dgs/garden/` |
| Quality pass — marching-cubes / NKSR if blobby | ⬜ | 2026-05-26 | only if v2 rejected |

### Phase 6 — full HunyuanWorld pipeline ⛔ (hardware-blocked)

| Task | Status | Target | Notes |
|---|---|---|---|
| Stand up vLLM server (Qwen3-VL-8B) | ⛔ | TBD | needs a separate GPU pool |
| WorldStereo 2.0 single-GPU load test | ⬜ | TBD | ~34GB fp16; drop `--fsdp` |
| WorldNav trajectory planning | ⛔ | TBD | blocked on vLLM |
| End-to-end panorama → navigable world | ⛔ | TBD | needs all of the above |

### Phase 7 — product concept: "Sekai Walk" ⬜

| Task | Status | Target | Notes |
|---|---|---|---|
| Lock name + tagline | ⬜ | 2026-05-26 | leading: **Sekai Walk** — "Walk anywhere that ever was" |
| One-page pitch README | ⬜ | 2026-05-30 | what it does, tech stack, demo links |
| Landing-page wireframe | ⬜ | 2026-06-06 | |
| Scene library — history / anime / real | ⬜ | 2026-06-13 | HunyuanWorld + Lyra + splatfacto outputs |
| VR build target (WebXR via three.js) | ⬜ | 2026-06-20 | meshes import cleanly; splats need a WebXR renderer |

---

## Immediate next action  ◀── start here

**Review the garden mesh v2** on <https://kitan-a.com/3dgs/garden/>
(dropdown → "mesh"). If it looks correct, flip Phase 5 row 3 to ✅. If
still blobby, the dials are at the top of
`garden-mesh-viewer/scripts/splat_to_mesh.py` (BBOX, OPA_MIN,
POISSON_DEPTH, DENSITY_PCT) — or escalate to marching-cubes / NKSR.

---

## Live links

- **Project intro (collaborators):** <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/project/>
- **kitan-a.com demos:** `/3dgs/plant/` · `/3dgs/plant-4k/` · `/3dgs/lyra/` ·
  `/3dgs/hyworld/` · `/3dgs/garden/` · `/3dgs/` (room) · `/3dgs/colmap/`
- **Repos:** [world-models](https://github.com/vladimiralbrekhtccr/world-models) ·
  [garden-mesh-viewer](https://github.com/vladimiralbrekhtccr/garden-mesh-viewer) ·
  [360-panorama-viewer](https://github.com/vladimiralbrekhtccr/360-panorama-viewer)

---

## How to add deadline tracking on GitHub (optional upgrades)

This `PLAN.md` covers most needs. For reminders / a board view:

- **Milestones** — repo → Issues → Milestones → "New milestone" has a
  **due date** field; group issues under it.
- **GitHub Projects (v2)** — repo → Projects → "New project" → **Table**
  or **Roadmap** layout, add a custom **Date** field. This is the
  closest thing to an online spreadsheet with deadlines. To drive it
  from the CLI too, run once: `gh auth refresh -s project`.
- **Issues** — turn any ⬜ row into an issue with
  `gh issue create --title "..." --milestone "..."`.

Recommendation: keep `PLAN.md` as the source of truth; add a Milestone
only for the **Phase 7 pitch** if you want a visible countdown.

---

## Decision log

- **2026-05-11** — Pick `witch_hat_atelier` as the first MultiPano scene.
- **2026-05-13** — Skip anime-stills-scrubbing for style references; use
  one Google-image style anchor + optional hand sketch.
- **2026-05-14** — Project framed publicly as *"Explorable World
  Models"*; topic label **neural rendering**. `/project/` intro published.
- **2026-05-17** — ChatGPT `gpt-image-2` UI won't produce real
  equirectangular panoramas; Codex's `image_gen` tool does. Switched.
- **2026-05-17** — MultiPano Step 5: `recon_3dgs.py` works after pointing
  `CUDA_HOME` at `~/miniconda3` (CUDA 12.8). 590k-gaussian scene deployed.
- **2026-05-18/19** — splatfacto-big plant demos (720p + 4K) deployed;
  Lyra 2.0 arc-orbit chapel recipe validated and documented in CLAUDE.md.
- **2026-05-19** — HunyuanWorld 2.0: built single-GPU `hyworld2` env,
  swept WorldMirror over 14 examples (14/14, 2.35GB VRAM), deployed 3
  scenes to `/3dgs/hyworld/`. Full WorldGen pipeline + HY-Pano-2 skipped
  — they need multi-GPU + a vLLM server.
- **2026-05-20** — garden 3DGS → Poisson mesh; published the standalone
  `garden-mesh-viewer` repo. Lesson: naive Poisson on raw splats is
  blobby — must filter by opacity/scale + crop before reconstructing.
- **2026-05-22** — Naming exploration for the product layer. Leading
  candidate **Sekai Walk** (immersion-first, content-agnostic — history
  / anime / real). "Sekai" = Japanese for "world", not anime-specific.
