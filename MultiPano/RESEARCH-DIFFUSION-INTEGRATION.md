# Diffusion / video-diffusion for MultiPano gap-filling

Working note · 2026-05-17 · drafted while user was AFK exploring World Labs.

## Why we care

The current MultiPano pipeline produces a fused depth-shell 3DGS scene
from 3 panoramas (deployed at `/3dgs-multipano/`). What it can't do:

- Fill **occluded regions** that no panorama saw (back of the chapel,
  ground behind a tree).
- Heal **per-pano depth artefacts** (the depth-skirts that ruined Path A).
- Reconcile **cross-pano inconsistencies** (the chapel roof drifting a
  tile between pano 1 and 3).

These are exactly the failure modes a strong 2D / video-diffusion prior
addresses. The 2026 literature has converged on a clear pattern.

## The canonical pattern, as of 2026

```
        sparse-view inputs (3 panoramas, our case)
                 │
                 ▼
   coarse 3D recon (sparse 3DGS or depth fusion)
                 │
                 ▼
   render a "fly-through video" from coarse 3D
                 │
                 ▼
   video diffusion **corrects** the video
       (fixes geometry, fills holes, sharpens textures)
                 │
                 ▼
   use the corrected frames as **dense supervision**
   for a second 3DGS optimization pass
                 │
                 ▼
   refined scene.ply
```

This is the "diffusion-as-3D-prior" loop. The video diffusion is doing
two jobs simultaneously: novel-view synthesis AND geometric consistency
checking (temporal coherence ≈ 3D consistency by construction).

## Landscape — papers worth knowing about

Ranked by relevance to our specific pipeline:

### Tier 1 — most directly applicable

1. **Lyra** (NVIDIA, Sept 2025) — [paper](https://arxiv.org/abs/2509.19296) ·
   [project](https://research.nvidia.com/labs/toronto-ai/lyra/) ·
   [HF](https://huggingface.co/papers/2509.19296).
   *Self-distills* implicit 3D knowledge in a pretrained video diffusion
   model into explicit 3DGS. No multi-view training data needed. SOTA on
   static + dynamic 3D scene generation. **This is the closest analogue
   to where we want our pipeline to go.**

2. **ReconX** (Aug 2024) — [project](https://liuff19.github.io/ReconX/).
   Sparse-view 3D reconstruction by encoding a coarse point cloud as a
   3D structure condition for a video diffusion model, which then
   synthesises 3D-consistent frames. Final scene is recovered via
   confidence-aware 3DGS optimization. **Almost exactly our problem
   statement.**

3. **WorldWarp** (Dec 2025) — [paper](https://arxiv.org/html/2512.19678).
   Spatio-Temporal Diffusion (ST-Diff) with a "fill-and-revise"
   objective specifically built for **filling occluded regions** with a
   spatio-temporal varying noise schedule. **Direct gap-filling.**

### Tier 2 — useful priors and tricks

4. **CAT3D** (Google, NeurIPS 2024) — [paper](https://arxiv.org/html/2405.10314v1).
   The classic multi-view diffusion model. Generates consistent novel
   views from sparse inputs. 3D self-attention. Decouples generation
   from reconstruction.

5. **GaussVideoDreamer** (Apr 2025) — Single-image 3D via video
   diffusion + Inconsistency-Aware Gaussian Splatting. *Detects 3D
   errors and ignores them.* Useful idea: a confidence weight on each
   Gaussian based on cross-view agreement.

6. **VideoScene** (Apr 2025) — distills video diffusion into 3D scenes
   in **one step**. Latency-optimised. Probably overkill for v1.

7. **Diff4Splat** (Nov 2025) — repurposes video diffusion for dynamic
   scene generation. Not relevant unless we add a temporal axis
   (4DGS).

8. **ReconFusion** (CVPR 2024) — diffusion prior regularising NeRF for
   sparse-view; precursor to most of the above.

9. **RI3D** (Mar 2025) — Repair & Inpainting diffusion priors for
   few-shot Gaussian Splatting. Heavy on the inpainting side.

10. **SplatDiff** (SIGGRAPH 2025) — pixel-splatting-guided video
    diffusion. Uses the splatting itself as conditioning. Beautiful
    bidirectionality.

11. **MVSplat360** (NeurIPS 2024) — feed-forward 360 scene synthesis
    from sparse views. Maybe applicable directly to our panorama
    setup.

12. **DreamScene360** (ECCV 2024) — already in our `3DGS/`
    notes as the Path B starting point. Now superseded by Lyra-class
    methods but still a relevant reference.

## How this maps to our pipeline

Our current state (`/3dgs-multipano/`): coarse depth-fusion 3DGS with
~1.2 M Gaussians and visible artefacts. We have 3 source panoramas, a
map, known camera poses.

### Three concrete integration plans, ordered by effort

**Plan A — Image-SDS refinement (easiest, ~1–2 days).**
Render N novel views from the coarse 3DGS at random nearby camera
positions. For each rendered view, pass through Stable Diffusion XL +
ControlNet (depth-conditioned) → get a "cleaned-up" version. Use those
cleaned-up images as new supervision for a second 3DGS pass. No video
model needed; just per-frame diffusion priors. The DreamScene360
recipe — well understood, runs on one H100, no exotic deps.

**Plan B — Video-SDS refinement (Lyra/ReconX style, ~1 week).**
Render a smooth flythrough trajectory from the coarse 3DGS (e.g.
along a path through pin 1 → pin 2 → pin 3). Pass through a video
diffusion model (CogVideoX-5B, Wan-2.1, or similar 2026 model) with
appropriate conditioning. The video diffusion enforces *temporal
coherence* on the rendered video, which is mathematically equivalent
to 3D consistency. Use the corrected frames as supervision. Heavier
GPU memory, longer training, but produces the cleanest result.

**Plan C — Panorama densification (the practical hack, ~half a day).**
Skip the SDS loop entirely. Use a 2D image diffusion + IP-Adapter to
*generate more panoramas* at intermediate map positions (e.g. midpoint
between pin 1 and pin 2), each conditioned on the surrounding existing
panoramas. Now we have 6–10 panoramas instead of 3, run the existing
depth-fusion → much denser coverage. This is what we should try
**first** because the rest of the pipeline doesn't change at all.

### Recommended first step

**Plan C, then graduate to Plan A.**

- Plan C is the smallest pipeline change for the largest immediate win
  (more views with the existing fusion code).
- Plan A is the right "real refinement" target once C has saturated
  what depth-fusion can do.
- Plan B / Lyra-class is research-grade — worth aiming for in the
  3-month horizon, not the 3-day horizon.

## Models to actually download

If we commit to Plan B (later), the candidate video diffusion models to
download to the cluster:

- **Wan-2.1** — Alibaba's open video diffusion, T2V & I2V variants.
  ~10 GB. Permissive license. Strong 2025–26 model.
- **CogVideoX-5B** — THUDM, open weights, well-documented.
- **HunyuanVideo** — Tencent, ~13 GB, T2V.
- **Mochi-1** — Genmo, open, ~10 GB, T2V.

For Plan A (image diffusion), we already have `diffusers` in the venv.
Stable Diffusion XL + ControlNet (depth) is a few GB total. Could be
running today.

## Open questions worth listing

1. **Does Lyra release inference code?** If so, we should just *use* it
   directly rather than rebuild the pattern. Their HF page suggests yes
   but needs verification.
2. **Can a video diffusion accept an equirectangular trajectory** as
   conditioning, or does it expect rectilinear video? Most are
   rectilinear → we'd need to slice the flythrough into perspective
   sequences, run the video model per direction, and re-stitch.
3. **What's the cost/quality curve** for trajectory length? A 24-frame
   flythrough through 3 pins vs. a 96-frame dense circle around the
   chapel — which gives the cleanest refined 3DGS per GPU-hour?
4. **Identity drift in the diffusion pass.** SDS-style refinement can
   "rewrite" the scene more than we want. GaussVideoDreamer's
   inconsistency-aware splatting is a candidate mitigation: each
   Gaussian carries a confidence; low-confidence ones get updated
   freely, high-confidence ones get held fixed.

## What I'd build next, if given an uninterrupted week

1. **Day 1–2**: Plan C (panorama densification via SDXL + IP-Adapter).
   Generate ~6 intermediate panoramas. Re-fuse. Deploy
   `/3dgs-multipano-v2/` and compare visually to `/3dgs-multipano/`.
2. **Day 3–4**: Plan A (image-SDS refinement on the coarse 3DGS).
   Render 200 novel views, diffuse each with depth-conditioned SDXL,
   train second 3DGS pass with those as supervision.
3. **Day 5**: Investigate Lyra's release. If usable, just run it on our
   inputs and compare to everything above.
4. **Day 6–7**: Build the actual `recon_video_sds.py` (Plan B) using
   Wan-2.1 as the video prior. This is the research contribution if it
   works.

## Pointers for the user (to discuss when back)

Two design decisions to make before writing any more code:

1. **Plan A vs Plan C as the first refinement step.**  Plan A gives the
   cleaner conceptual story (closer to published methods). Plan C gives
   a faster immediate win with no architectural change.

2. **Whether to commit to a specific video diffusion model now.**
   Picking Wan-2.1 locks us into ~10 GB of weights and ~3 days of
   integration work. Worth doing if Lyra's inference isn't available.
   Otherwise, defer until the simpler plans saturate.
