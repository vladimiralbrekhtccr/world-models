# Ryōtenkai — WACV 2027 paper · full project brief

A comprehensive context document for anyone (collaborator or LLM)
helping me execute the work plan in `tracker.csv`. Read this once,
then jump to whichever CSV row I'm asking about.

---

## 1. The bigger picture

The umbrella project is **world-models** — an exploration repo
([github.com/vladimiralbrekhtccr/world-models](https://github.com/vladimiralbrekhtccr/world-models))
whose long-term goal mirrors what World Labs, Marble, and a growing
cluster of academic groups are chasing: **"single image or prompt →
explorable 3D world"** — a system anyone can hand a photo, a sentence,
or a 360° pano and receive back a walkable scene they can fly
through in a browser or VR headset.

3D Gaussian Splatting (3DGS; Kerbl et al., SIGGRAPH 2023) is the
rendering substrate that finally makes this practical: real-time,
explicit-representation, web-deployable. NeRF couldn't be streamed or
edited; 3DGS can. That's why the last 2 years have produced an
explosion of papers on splat *capture*, *generation*, *meshing*, and
*deployment* — and why there's still a lot of unclaimed ground at the
**applied / systems / web-deployment** end of the field.

That's the gap this paper aims at: **practical pipelines that take
real-world inputs and produce real-world-usable artifacts on
real-world hardware** (single H100, web-renderable output).

## 2. Where I am right now (late May 2026)

After roughly 4 months of exploration, the lab side is mature:

- **All five "image/prompt → 3DGS" pipelines that fit on one GPU are
  installed, validated, and deployed as live demos.** (Detailed list
  in §3 below.)
- **A teaching course is being polished** at
  [kitan-a.com/3dgs/learn/](https://kitan-a.com/3dgs/learn/) — seven
  modules with interactive widgets and a published explainer-style
  HTML page (Bartosz Ciechanowski / Red Blob Games visual lineage).
- **A naming / brand identity is locked**: the working name is
  **Ryōtenkai** (領域展開, "domain expansion" — a Jujutsu Kaisen
  reference; see §7 below).
- **A project plan + tracker** lives in the repo:
  [`PLAN.md`](https://github.com/vladimiralbrekhtccr/world-models/blob/main/PLAN.md).

What's left, and the reason for this brief: **convert the lab
experience into one focused, defensible academic paper, accepted at
WACV 2027.**

## 3. The five working pipelines

These are the experimental assets the paper will build on. Pick
whichever your task most needs.

### 3.1  splatfacto-big — classical capture
- **What:** phone video → ffmpeg-extracted frames → COLMAP SfM →
  splatfacto-big (nerfstudio) 30k iterations → `.ply` → ksplat web
  bundle.
- **Why it matters for the paper:** the **classical baseline**.
  Every comparison in the paper will reference it as "what you get
  with the well-trodden capture pipeline."
- **Live demos:** [kitan-a.com/3dgs/plant/](https://kitan-a.com/3dgs/plant/) (720p) ·
  [kitan-a.com/3dgs/plant-4k/](https://kitan-a.com/3dgs/plant-4k/) (4K).
- **Time cost:** ~30 min on one H100 per scene (COLMAP is the
  bottleneck, especially at 4K).
- **Code:** `MultiPano/run_nerfstudio_big.sh` in the repo.

### 3.2  MultiPano — multi-panorama → real-geometry 3DGS
- **What:** 3 hand-placed overlapping 360° panoramas + camera pin
  positions → `recon_3dgs.py` (gsplat with equirectangular camera
  model) → walkable splat.
- **Why it matters:** demonstrates 3DGS works from a sparse-pano
  capture rig — the data modality is panoramic image, not photos.
- **Live demos:** [/3dgs-multipano-mv/](https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs-multipano-mv/)
  (real multi-view) · [/3dgs-multipano/](https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs-multipano/)
  (depth-fusion baseline).

### 3.3  Lyra 2.0 — single image → diffusion → splat
- **What:** NVIDIA's Lyra 2.0 (April 2026, Apache 2.0, public weights).
  Take one anime panorama → video-diffusion generates a 481-frame
  orbit → gsplat photometric+SSIM optimization on that video using
  VIPE-derived poses → clean splat.
- **Why it matters:** the **generative-pipeline reference**. Proves
  diffusion can imagine a coherent walkable world from one 2D image.
- **Live demo:** [kitan-a.com/3dgs/lyra/](https://kitan-a.com/3dgs/lyra/)
  (chapel scene, 1.5M optimized gaussians).
- **Key lesson:** Lyra's own Step-2 feed-forward recon is a soft
  transparent cloud — *don't ship it*. Use Lyra only to generate the
  video, then run real gsplat optimization. Documented as the
  "Lyra recipe" in `CLAUDE.md` §D.

### 3.4  HunyuanWorld 2.0 WorldMirror — feed-forward recon
- **What:** Tencent's HY-World 2.0 WorldMirror-2 (~1.2B parameter
  transformer in the DUSt3R/MASt3R/VGGT lineage). 2–32 images → splat
  in **~1 second** on one GPU at **2.35 GB VRAM**. No COLMAP needed
  — predicts camera poses internally.
- **Why it matters:** the **feed-forward reference**. Newest paradigm,
  highest speed:quality ratio for short-input scenes.
- **Live demo:** [kitan-a.com/3dgs/hyworld/](https://kitan-a.com/3dgs/hyworld/)
  (3 scenes: Statue Face, Small Room, Tree+Building).
- **Tested at scale:** 14 scenes swept, 14/14 succeeded, no OOM.
- **Skipped (won't fit one GPU):** the rest of the HY-World pipeline
  — HY-Pano-2 (~80B params) and the WorldGen stages (need vLLM server
  + 8-GPU FSDP). See [`HUNYUANWORLD_PIPELINE.md`](https://github.com/vladimiralbrekhtccr/world-models/blob/main/HUNYUANWORLD_PIPELINE.md)
  for the full 5-stage breakdown.

### 3.5  Garden splat → mesh
- **What:** take a 3DGS scene → opacity filter (>0.5) → scale filter
  (drop background-sphere blobs) → bounding-box crop → estimate
  normals → Poisson reconstruction depth 11 → largest-connected-
  component pruning → per-vertex colour from nearest splat → `.glb` or
  `.ply` mesh.
- **Why it matters:** **this is the most paper-ready experimental
  asset.** Mesh extraction is the bridge between splats and the
  broader 3D-tools ecosystem (Unity, Unreal, Blender). The recipe
  isn't well-evaluated in the literature.
- **Live demo:** [kitan-a.com/3dgs/garden/](https://kitan-a.com/3dgs/garden/)
  (splat / mesh / overlay toggle).
- **Standalone repo:** [github.com/vladimiralbrekhtccr/garden-mesh-viewer](https://github.com/vladimiralbrekhtccr/garden-mesh-viewer)
  — already has README, scripts, env, deploy.

## 4. The paper plan

### 4.1  Venue choice — why WACV

WACV ([wacv.thecvf.com](https://wacv.thecvf.com)) is the IEEE/CVF
**Applications of Computer Vision** conference. Compared to CVPR /
ICCV / 3DV, WACV deliberately favours **applied / systems / engineered
contributions** over theoretical novelty. Looking at the WACV 2026
accepted-paper list (3D Superquadric Splatting, ForestSplats,
MagicDrive3D, Gaussian Swaying, GDoFS, STRinGS, RapidMV…) every one
is the same shape: *3DGS + one focused technical delta or
application + benchmark numbers + ablations.*

That's the bar. It's reachable with what I already have.

### 4.2  Timeline

| Date | Event |
|---|---|
| **Aug 20, 2026** | Round 2 paper **registration** — title + abstract + authors + keywords locked in the OpenReview-style system. *No PDF yet, but you cannot submit later if you missed this.* |
| **Aug 27, 2026** | Round 2 paper **submission** — PDF + supplementary uploaded by 23:59 AoE. |
| Oct 9, 2026 | Reviews + decisions back. |
| Nov 1, 2026 | Camera-ready due. |
| Jan 5–9, 2027 | Conference, Disney Springs, Florida. |

**Internal soft milestone:** **Jul 20, 2026** — full first draft
circulated to readers. Gives a full month of polish before Aug 27.

### 4.3  The three candidate angles

These are written for non-specialists; the actual paper will need to
be much more technical.

**Angle A1 — Splat → Mesh for Web Delivery** ⭐ *recommended*
- **Problem statement:** 3DGS scenes are great to look at but
  awkward to deploy — `.ply` files are large, splat renderers aren't
  installed everywhere, and the splat representation doesn't compose
  with game-engine pipelines (Unity, Unreal, Blender). Converting a
  splat to a polygon mesh is a known need but poorly evaluated; the
  trade-offs between Poisson, marching cubes, TSDF fusion, SuGaR,
  and 2DGS are scattered across papers.
- **Contribution:** a practical, reproducible pipeline that takes
  *arbitrary 3DGS scenes* (not just specially-trained ones) and
  produces web-deployable meshes at a fraction of the splat's size,
  with quantified visual-fidelity loss. Plus an ablation that shows
  which filters matter (opacity, scale, bbox, normal orientation,
  Poisson depth, LCC).
- **Empirical setup:** Mip-NeRF 360 (7 scenes) as the standard
  benchmark. Compare against SuGaR, 2D Gaussian Splatting, raw
  Poisson, marching cubes. Metrics: PSNR / SSIM / LPIPS *of mesh
  renders vs original splat renders*, Chamfer distance, mesh size,
  triangle count, web load time.
- **Why this is the realistic pick:** ~60% of the experiments are
  done — the garden-mesh-viewer repo has the pipeline working on
  one scene with documented filtering recipe. Need to extend to
  7 scenes + add baselines + ablations.

**Angle A2 — Unified Benchmark for Explorable Scene Synthesis**
- **Problem statement:** there's no apples-to-apples comparison of
  splatfacto vs WorldMirror vs Lyra vs HunyuanWorld on the same
  input set. Each paper reports its own numbers on its own benchmark.
- **Contribution:** a unified evaluation harness + a new
  "explorability" metric: how does scene quality degrade as the
  camera moves away from the input-view positions? (proxies: render
  quality vs angular distance, hole percentage, novel-view drift).
- **Why this is harder to land:** benchmark/eval papers have a higher
  bar — the new metric has to clearly improve over what exists, or
  the paper has nowhere to go.

**Angle A3 — TENKAI End-to-End System**
- **Problem statement:** no existing pipeline ties together a
  feed-forward initializer (WorldMirror), a generative refiner
  (Lyra-style diffusion), per-scene optimization (gsplat), and
  mesh extraction into a single deployable system.
- **Contribution:** one-pipeline system + system-level evaluation.
- **Why this is risky:** 5-stage system in 8 pages tends to read
  shallow. Reviewers complain "you stapled things together."

**Recommendation: pick A1 by Jun 15.** Most realistic. Most
experiments done. Cleanest reviewer story.

### 4.4  Title

Working title: **Ryōtenkai: One-Shot Domain Expansion for Walkable
3D Worlds** — the JJK codename + a tagline. For an A1-angle paper,
the tagline will probably tighten to something more literal in the
final submission (e.g. *"Practical Mesh Extraction from 3D Gaussian
Splats for Web Delivery"*), keeping the codename as the brand.

## 5. The 9 stages, in detail

Each maps to a section of `tracker.csv` with concrete tasks. Below is
the *intent* of each stage — what success looks like, what failure
modes to avoid.

### Stage 0 — Kickoff (May 27 – Jun 3)
**Intent:** remove every friction that would slow down the actual
research. Repo set up, template downloaded, reference manager
installed, calendar slots blocked. Cheap, mechanical, do it now.
**Done when:** you can write a paragraph of the paper today without
hunting for any tool.

### Stage 1 — Foundations (May 27 – Jun 10, parallel with Stage 0)
**Intent:** internalize the 6 papers that ground everything else.
Read **3DGS (Kerbl 2023), NeRF (Mildenhall 2020), Mip-NeRF 360
(Barron 2022), DUSt3R / MASt3R / VGGT, SuGaR (Guédon 2024), 2DGS
(Huang 2024)**. Take physical notes. The §3+4 of the 3DGS paper
should be re-readable in your head.
**Done when:** you can explain *each paper's contribution in two
sentences without looking it up*, and you can draw the splat
rasterization pipeline on a whiteboard.
**Failure mode:** skim-reading. Each paper deserves 2–3 hours.

### Stage 2 — Literature deep dive + angle lock (Jun 3 – Jun 17)
**Intent:** read the most relevant 15–20 recent papers, including
8–10 from WACV 2026, then **pick the angle (A1 / A2 / A3) and lock
the title + 200-word abstract on Jun 17**. This is the highest-leverage
decision in the whole plan.
**Done when:** you can articulate (a) which gap in the literature
your paper fills, (b) what your 3 baseline comparators will be,
and (c) what metric will tell reviewers you're better.
**Failure mode:** putting off the angle decision. Every later stage
compresses if Jun 15 slips. *Decide on the 15th even if you don't
feel ready — you can pivot once if reviewers in §3 say no, but you
cannot waffle for 3 weeks.*

### Stage 3 — Baselines + evaluation harness (Jun 17 – Jul 1)
**Intent:** reproduce your 3 baseline methods on 1 scene each,
verify your evaluation pipeline matches published numbers within
±0.5 (PSNR) / 0.005 (SSIM). Build the evaluation harness once and
re-use it for every later experiment.
**Done when:** running `eval.py --method baseline_N --scene garden`
prints a reproducible row of metrics that matches the baseline paper.
**Failure mode:** rushing past this stage. Bad baselines = paper
gets rejected for "unfair comparison." Spend the full 2 weeks here.

### Stage 4 — Method + experiments (Jul 1 – Jul 8)
**Intent:** implement your method on top of the baselines, run on
the full benchmark (Mip-NeRF 360 = 7 scenes), collect raw numbers
into a CSV. *Don't analyze yet — just collect.*
**Done when:** every (method × scene × metric) cell of the
results table has a number in it.
**Failure mode:** writing the paper before you have all the
numbers. Resist.

### Stage 5 — Ablations + figures (Jul 8 – Jul 15)
**Intent:** ablate 3–5 components of your method, build the
qualitative-comparison figure, build the main results table, build
the teaser image. Figures should be 80% of the paper's
visual-real-estate; reviewers skim figures before reading body.
**Done when:** every figure has a one-sentence caption that *alone*
makes the contribution clear.
**Failure mode:** drawing figures in PowerPoint at the end. Use
matplotlib + LaTeX from the start so they're editable.

### Stage 6 — First full draft (Jul 8 – Jul 20)
**Intent:** every section drafted, hooked up to figures and tables.
The draft can be ugly — clarity > prose polish.
**Done when:** you can hand the PDF to a stranger and they
understand the contribution without you in the room.
**Failure mode:** writing related work last. Write it concurrently
with method (Jul 12–14) so your method section can cite cleanly.

### Stage 7 — Internal review + polish (Jul 20 – Aug 19)
**Intent:** 2–3 readers, iterate on figures, run any final
reviewer-flagged ablations, polish abstract to final form,
prepare supplementary material.
**Done when:** abstract is one paragraph you can recite verbatim,
and every figure caption stands alone.

### Stage 8 — Registration + submission (Aug 19 – Aug 27)
**Intent:** **Aug 20**: register the paper (title + abstract +
authors + keywords). **Aug 27**: submit the PDF + supplementary by
23:59 AoE.
**Done when:** OpenReview confirmation email in inbox.
**Failure mode:** missing Aug 20 registration → cannot submit on
Aug 27. *No exceptions, no grace period.* Set 3 calendar reminders
+ 1 phone alarm for Aug 20.

## 6. Hard constraints (the experimental environment)

These are real-world constraints baked into every task. Don't suggest
work that violates them.

- **One GPU only**, specifically GPU 6 on `node001` of the cluster
  (other GPUs are used by other workloads). Set
  `CUDA_VISIBLE_DEVICES=6` for everything.
- **No SLURM**. The user holds an interactive allocation;
  `ssh node001` and run commands there.
- **No home-directory writes** — `/home` is 99% full. All caches
  (HF, pip, torch, conda) are redirected to `/scratch`.
- **Detach long jobs properly** — `nohup setsid bash <abs-path> >
  <abs-log> 2>&1 < /dev/null &`. ssh-session teardown can SIGHUP a
  COLMAP grandchild otherwise.
- **Deployment is to kitan-a.com** via Caddy on the `foggen` host.
  Static routes: `/var/www/<name>/` + a `handle_path /<name>/*`
  block in `/etc/caddy/Caddyfile`. `scp` files to deploy.

## 7. Naming / branding context

The codename **Ryōtenkai** (領域展開, "domain expansion") comes
from Jujutsu Kaisen, where sorcerers create an inner world (their
*domain*) that overlays reality, drag opponents into it, and apply
their rules inside. The metaphor maps cleanly to what the system
does: input goes in → 3D world unfolds around the viewer → they
explore it in WebXR.

In the paper:
- **Title:** keep the codename + a more literal subtitle.
- **Abstract opener:** the JJK metaphor as a rhetorical hook
  (*"In jujutsu sorcery, a 'domain expansion' is the moment a
  sorcerer unfolds an inner world that overlays reality. We borrow
  the term as a metaphor for…"*). Reviewers remember papers with
  hooks.
- **Body text:** mostly drop the metaphor, use standard CV
  vocabulary (*"scene generation"*, *"novel-view synthesis"*,
  *"mesh extraction"*).

## 8. How to help me (LLM guidance)

When I paste this brief in and then ask for help on a CSV row:

- **Anchor everything to the single-H100 stack.** Don't propose
  experiments that need multi-GPU or external services I haven't
  already set up. If a stage genuinely requires more compute, say
  so explicitly so I can decide whether to scope it differently.
- **Stick to the picked angle.** If I've said "I'm going with A1,"
  don't pull me toward A3. If I haven't picked yet (before Jun 15),
  help me decide — don't waffle.
- **WACV is an *applications* conference.** Favour practical,
  empirical, deployment-focused advice over theoretical-novelty
  framing. Reviewers want a clean systems story + ablations on
  standard benchmarks.
- **Concrete next steps over high-level theorizing.** The plan is
  already made. When I ask for help on Stage N Task M, the answer
  should be *"do these three things in this order,"* not *"first,
  consider whether…"*.
- **Use the names I use.** Methods are *splatfacto*, *Lyra*,
  *WorldMirror*. Scenes are the Mip-NeRF 360 names (*garden*,
  *kitchen*, *bicycle*, *room*, *counter*, *stump*, *bonsai*).
  Codename is *Ryōtenkai*. Don't invent new vocabulary.
- **Match my voice.** Terse, direct, no hand-holding. If I'm
  confused I'll ask.

The CSV is the work plan. This brief is the context. Help me
execute the nearest todo row.
