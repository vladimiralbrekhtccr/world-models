# PLAN.md — picking back up

> If you've been away from this repo for a few days, **start here.** This
> file is the resume-doc — it tells you where we are, what the immediate
> next action is, and where every piece lives.

For the outward-facing *"what is this project"* intro, see
<https://vladimiralbrekhtccr.github.io/360-panorama-viewer/project/>.
That page is for collaborators; this file is for you.

---

## What changed while you were AFK (2026-05-17 session)

**TWO walkable MultiPano scenes exist now (compare them side-by-side):**

- 🟡 Depth-fusion baseline: <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs-multipano/>
- 🟢 **Real multi-view 3DGS**: <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs-multipano-mv/>
  (this is the artifact the entire pipeline was designed to produce —
  proper parallax-driven Gaussians, ~590k of them, 38 MB, trained in
  ~14 s on one H100 once gsplat finally compiled).

**Path from one panorama to map to 3 panoramas to walkable 3DGS now
runs end-to-end:**

- 3 Codex-generated panoramas of `witch_hat_atelier` (Step 4 done).
- Camera pins placed via `place-pins.html`; `poses.json` saved.
- `recon_fusion.py` written + run on node001 → 1.18 M-Gaussian PLY
  (deployed as `/3dgs-multipano/`, the depth-fusion baseline).
- `recon_3dgs.py` finally runs too — after pointing `CUDA_HOME` at
  miniconda3's 12.8 toolkit, gsplat JIT-compiled and the **real
  multi-view 3DGS** training produced a 590k-Gaussian PLY in ~94s
  (deployed as `/3dgs-multipano-mv/`). The working incantation is
  preserved in `MultiPano/run_recon_3dgs.sh`.

**A diffusion-refinement plan is documented** in
`MultiPano/RESEARCH-DIFFUSION-INTEGRATION.md`. The headline find: NVIDIA
released **Lyra 2.0** (April 2026, Apache-2.0, code + weights public)
which does basically what MultiPano wants to be — image + camera
trajectory + captions → explorable 3D Gaussians, via video-diffusion
self-distillation. Installation kicked off in background on node001;
~45-60 min total. See `MultiPano/setup_lyra.sh`.

**Things to look at first when you sit down:**

1. The deployed demo: <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs-multipano/>
2. The research note: `MultiPano/RESEARCH-DIFFUSION-INTEGRATION.md`
3. The state of the Lyra install:
   `tail -50 /scratch/vladimir_albrekht/projects/world-models/MultiPano/setup_lyra.log`

---

## Current state — May 2026

- **Path A** (single panorama → 3DGS) is done and deployed: splat,
  walkable mesh, and the interactive explainer page (with a 6-s rendered
  training video) all live on `360-panorama-viewer` GH Pages.
- **MultiPano** (multi-panorama → real-geometry 3DGS, the next big
  experiment) — concept doc + reusable gpt-image-2 prompts + input
  folder convention are all ready, but **no actual map or panoramas
  have been generated yet**. This is the next unblock.
- **Reading PDF app** is parked at a known highlight-render bug.
  See `/scratch/vladimir_albrekht/projects/reading/README.md`.

---

## Immediate next action  ◀── start here

**Generate the first map** for the `witch_hat_atelier` scene.

Decided last session: **don't scrub the anime for stills** — too much
effort. The lighter alternative:

```
1. Google "Witch Hat Atelier" → grab ONE wallpaper / key visual / fan-art
   that captures the style.        ← style anchor

2. (optional) 30-second hand sketch of the layout you want — blobs and
   lines labelled "chapel / tree / path / wall". Photograph it.

3. Drop both files into MultiPano/input/witch_hat_atelier/ via the
   inputs gallery:
       https://kitan-a.com/ide/proxy/8765/inputs.html

4. Open ChatGPT (gpt-image-2). Paste the Step 1 prompt from
   MultiPano/README.md. Attach the style image (+ sketch if present).
   It will infer the layout and produce a top-down map.

5. Save the result as MultiPano/input/witch_hat_atelier/map.png.
```

Once `map.png` exists, the rest of the pipeline can move.

---

## The full pipeline — where each step sits

```
┌──── Step 1 ────┐  ┌── Step 2 ──┐  ┌── Step 3 ──┐  ┌── Step 4 ──┐  ┌── Step 5 ──┐  ┌── Step 6 ──┐
│ Style + layout │→ │    Map     │→ │   Camera   │→ │ 3 Panoramas│→ │ Multi-view │→ │   Browser  │
│ references     │  │ gpt-image-2│  │ pin coords │  │ gpt-image-2│  │ 3DGS train │  │ walkthrough│
│   ✅ done     │  │  ✅ done    │  │  ✅ done   │  │  ✅ done    │  │  ✅ done    │  │  ✅ TWO    │
└────────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘
   Google + sketch    in ChatGPT      place-pins.html  via Codex's       recon_3dgs.py     /3dgs-multipano/
   (no anime          → map.png       + pano-prompts   image_gen tool    (real MV 3DGS     +/3dgs-multipano-mv/
   scrubbing)                         → poses.json     (real equirect    via gsplat        on panorama-viewer
                                                       — not ChatGPT     + DefaultStrategy) Pages
                                                       UI)               + recon_fusion.py
                                                                         (depth fusion)
```

**Step 5 (gsplat) finally works.** It needed `CUDA_HOME` pointing at
`~/miniconda3` (which has CUDA 12.8 + dev headers) and the
`LIBRARY_PATH`/`LD_LIBRARY_PATH`/`CPATH` triad set to the miniconda
lib + targets/x86_64-linux dirs. `MultiPano/run_recon_3dgs.sh` has the
incantation. After that gsplat JIT-compiles in ~80 s once per session,
then training is fast: 5000 iters in ~14 s on one H100, growing 20k →
590k Gaussians with `DefaultStrategy` adaptive density control.

`recon_fusion.py` (depth-fusion) is kept as a complementary baseline
for comparison — it's still useful when no GPU with CUDA libs is
available, or as a sanity-check oracle.

### Step-by-step detail

1. **Style + layout references** — user-driven, no code. Just put the
   files in `MultiPano/input/witch_hat_atelier/`.
2. **Generate map** — user pastes the Step 1 prompt + attachments into
   ChatGPT, saves `map.png`. The prompt template lives in
   `MultiPano/README.md` and the served version at
   <https://kitan-a.com/ide/proxy/8765/>.
3. **Place camera pins** — script `place_pins.py` (planned, ~30 lines):
   opens the map in a clicker tool, you click 3 positions, it saves
   `poses.json` with `{px, py, yaw_deg}` per pin + a declared
   `scale_m_per_px` so coords convert to metres.
4. **Generate panoramas** — for each pin, paste the Step 2 prompt from
   `MultiPano/README.md` into ChatGPT with the map + the previous
   panorama as image references. Save as `pano_1.png`, `pano_2.png`,
   `pano_3.png`. Eyeball: do shared landmarks look like the same
   landmarks? If yes, continue. If no, iterate the prompt or swap the
   generator before any 3D code is written.
5. **Train multi-view 3DGS** — script `recon_3dgs.py` (planned): reads
   `poses.json` + the 3 panoramas, uses an equirectangular camera model
   with `gsplat`, photometric loss + optional SDS regularizer against a
   2D diffusion prior (this is the "neural rendering proper" piece —
   fills the holes between viewpoints). Outputs one `scene.ply`.
6. **Browser walkthrough** — drop `scene.ply` into a copy of
   `360-panorama-viewer/3dgs/` (or `/3dgs-mesh/`). Both viewers already
   work; just need the new asset and a camera path through the 3 pins.

---

## Subprojects — quick status

| Folder | What it is | Status | Last move |
|---|---|---|---|
| `3DGS/` | Path A: single-panorama → splat / mesh | ✅ done | training-video renderer added |
| `MultiPano/` | Path B: multi-panorama → real 3DGS | ▶ active, blocked on Step 1 | concept doc + prompts + inputs gallery ready; no map yet |
| `reading/` (separate folder) | personal PDF reader | ⏸ paused | known highlight-render race; see its README |
| `360-panorama-viewer` repo | live demos + project intro | ✅ deployed | `/project/` intro page published |
| `foggen/AI_github/skills/` | Codex skill `address-doc-annotations` | ✅ committed | applies pasted annotation JSON |

---

## Quick links

- **Outward-facing project intro (for collaborators):**
  <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/project/>
- **Live demos:**
  - splat — <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs/>
  - mesh — <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs-mesh/>
  - explainer — <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs-explainer/>
  - annotation pattern — <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/annotation-workflow/>
- **MultiPano concept doc** (read this first when re-entering MultiPano):
  [`MultiPano/README.md`](MultiPano/README.md) — same content served
  interactively at <https://kitan-a.com/ide/proxy/8765/>.
- **Inputs gallery** (drag-drop scene references):
  <https://kitan-a.com/ide/proxy/8765/inputs.html>
- **GitHub repo:**
  <https://github.com/vladimiralbrekhtccr/world-models>

---

## When to update this file

- A pipeline step gets started or finished → flip its `⬜` to `▶` or `✅`.
- You context-switch back in → re-read **Immediate next action**.
- A research decision changes (scene anchor, aesthetic register, 4DGS
  in or out, browser vs standalone) → add a one-line entry under a
  new **Decision log** section at the bottom.

---

## Decision log

- **2026-05-11** — Pick `witch_hat_atelier` as the first MultiPano scene.
- **2026-05-13** — Skip anime-stills-scrubbing for Step 1 references;
  use one Google-image style anchor + optional hand sketch instead.
- **2026-05-14** — Project framing publicly named *"Explorable World
  Models"*. Topic label: **neural rendering** (CV + CG + generative ML).
- **2026-05-14** — `/project/` intro page published for collaborators.
- **2026-05-17** — ChatGPT (`gpt-image-2` via the UI) won't produce real
  equirectangular panoramas — only 2:1 wide-angle landscapes. Codex's
  built-in `image_gen` tool *does* produce real equirectangular projection
  (`View Image` → context → `image_gen` referencing the visible
  reference). Switched panorama generation to that path.
- **2026-05-17** — Step 4 done: 3 Codex-generated panoramas of the
  witch_hat_atelier scene, chapel framed right / centre / left to give
  real parallax between viewpoints.
- **2026-05-17** — Step 5 first cut as a fallback: gsplat wouldn't
  JIT-compile, so wrote `recon_fusion.py` — pure-PyTorch depth-fusion
  of the 3 panoramas. Output: `scene.ply` (1.18 M Gaussians, 77 MB),
  deployed as `/3dgs-multipano/`.
- **2026-05-17** — Step 5 PROPER: `recon_3dgs.py` works after pointing
  `CUDA_HOME` at `~/miniconda3` (CUDA 12.8 + dev headers were present
  all along, the conda install command's "InvalidSpec" error misled
  me — the headers under `.../targets/x86_64-linux/include/` already
  worked). gsplat JIT-compiles in ~80 s, then trains in ~14 s for 5000
  iters. Real multi-view scene: 590k Gaussians, 38 MB, deployed as
  `/3dgs-multipano-mv/`.
- **2026-05-17** — Started installation of NVIDIA's **Lyra 2.0** in the
  background on node001 (~45–60 min) — Apache-2.0 feed-forward video-
  diffusion 3D scene generator that could replace large parts of the
  hand-rolled MultiPano pipeline. See `MultiPano/setup_lyra.sh` and
  `MultiPano/RESEARCH-DIFFUSION-INTEGRATION.md`.
