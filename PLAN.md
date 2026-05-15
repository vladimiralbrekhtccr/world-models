# PLAN.md — picking back up

> If you've been away from this repo for a few days, **start here.** This
> file is the resume-doc — it tells you where we are, what the immediate
> next action is, and where every piece lives.

For the outward-facing *"what is this project"* intro, see
<https://vladimiralbrekhtccr.github.io/360-panorama-viewer/project/>.
That page is for collaborators; this file is for you.

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
│  ⬜ active     │  │  ⬜ blocked │  │  ⬜ blocked│  │  ⬜ blocked │  │  ⬜ blocked │  │  ⬜ blocked │
└────────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘
   you do it          you click it    place_pins.py    you click it     recon_3dgs.py    Three.js
   (Google + sketch)  in ChatGPT      (planned)        in ChatGPT, ×3   (planned, uses   (template from
                                                                       gsplat + SDS)    /3dgs-mesh/)
```

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
