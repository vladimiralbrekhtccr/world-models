# Project scope + tracker spec

A short brief describing the project, the paper, and what the
accompanying `tracker.csv` is for. Paste this into any LLM (Claude,
ChatGPT, etc.) when you want them to help with a task — they'll have
the context to be useful immediately.

---

## Project

**World-models** — an exploration repo that builds toward the goal:
*"single image or prompt → explorable 3D world."*

Current status: I've stood up and validated five 3DGS-adjacent pipelines
on one H100 (single GPU, no SLURM):

- **splatfacto-big** — classical capture: phone video → 3DGS via COLMAP
  + 30k gsplat iterations. Live demo: kitan-a.com/3dgs/plant-4k/
- **MultiPano** — multi-panorama → 3DGS. Live: kitan-a.com/3dgs-multipano-mv/
- **Lyra 2.0** (NVIDIA) — single image → diffusion video → splat.
  Live: kitan-a.com/3dgs/lyra/
- **HunyuanWorld 2.0 WorldMirror** (Tencent) — multi-view → splat in
  ~1 second feed-forward, 2.35 GB VRAM. Live: kitan-a.com/3dgs/hyworld/
- **Garden splat → mesh** — Poisson reconstruction with opacity/scale/
  bbox filtering. Live: kitan-a.com/3dgs/garden/ · separate repo:
  github.com/vladimiralbrekhtccr/garden-mesh-viewer

There's also a course/explainer being polished at kitan-a.com/3dgs/learn/
(seven modules + interactive widgets) and a project plan tracked in
[`PLAN.md`](https://github.com/vladimiralbrekhtccr/world-models/blob/main/PLAN.md)
of the repo.

## The paper

**Target:** WACV 2027 (Winter Conference on Applications of Computer
Vision) — Round 2 submission **Aug 27, 2026**, registration Aug 20,
conference Jan 5–9, 2027, Disney Springs FL.

**Working title:** *Ryōtenkai: One-Shot Domain Expansion for Walkable
3D Worlds* (codename — the JJK "domain expansion" 領域展開 metaphor
for unfolding a 3DGS world from a single input).

**Three candidate angles, one to be picked by Jun 15:**

- **A1 — Splat → Mesh for Web Delivery.** Practical engineering paper.
  Take Mip-NeRF 360 scenes, extract web-deployable meshes with
  opacity/scale/bbox filtering + Poisson + LCC pruning. Compare to SuGaR,
  2DGS, plain Poisson, marching cubes. Metrics: PSNR/SSIM/LPIPS,
  Chamfer, file size, web load time. *Most realistic for a first paper;
  60% of experiments already done in `garden-mesh-viewer`.*
- **A2 — Unified Benchmark.** Compare splatfacto vs WorldMirror vs Lyra
  on the same inputs. Propose a new "explorability" metric (e.g. quality
  vs angular distance from input views, hole percentage, drift).
- **A3 — TENKAI End-to-End System.** Single image → WorldMirror init →
  Lyra refinement → splat optimization → mesh extraction → web deploy.
  Most ambitious; risk of being shallow on each component.

**Recommendation: A1.** WACV is "Applications of CV" — practical
engineering with clear deployment story + measurable metrics fits the
venue best, and most of the empirical work is already done.

**Internal first-draft milestone:** Jul 20, 2026 (well before Aug 27).

## The CSV (`tracker.csv`)

A flat task list for the ~13 weeks between today (May 27, 2026) and the
WACV Round 2 submission (Aug 27, 2026). 43 tasks across 9 stages.

**Schema:**

| Column | Meaning |
|---|---|
| `Stage` | `0 - Kickoff`, `1 - Foundations`, `2 - Lit review`, `3 - Baselines`, `4 - Method + experiments`, `5 - Ablations + figures`, `6 - First draft`, `7 - Polish`, `8 - Submission` |
| `Task` | Short imperative title |
| `Description` | One line of additional context |
| `Target` | ISO date `YYYY-MM-DD` (sortable in any spreadsheet) |
| `Status` | `todo` (default), `doing`, `done`, `blocked` |
| `Notes` | Free text; `HARD DEADLINE` flagged for Aug 20 + Aug 27 rows |

**Cadence assumed:**
- Stages 0 + 1 run **in parallel** during Week 1 (setup + foundational reading).
- Lit review + angle lock finishes by Jun 17 — **angle decision lands Jun 15** (most leverage; don't slip).
- Baselines reproduced by Jul 1.
- Method + experiments done by Jul 8.
- First full draft by Jul 20 (matches the entry in my Google Sheet).
- Polish + reviewer feedback through August.
- Aug 20 registration, Aug 27 submission.

## How to help me (if you're an LLM reading this)

When I ask for help on a task from `tracker.csv`, you have all the
context you need from this file. Be aware that:

- **The paper isn't written yet**; we're in the planning + foundations
  phase. Help should usually be *concrete next-step* help (read this,
  draft that, structure these), not high-level theorizing.
- **I have a working stack** (single H100, validated pipelines). When
  suggesting experiments, anchor to what's already implemented. Don't
  propose anything that requires multi-GPU or external services I
  haven't already set up.
- **Stick to the picked angle.** If I've said "going with A1," don't
  pull me toward A3. If I haven't picked yet, help me pick.
- **WACV is an *applications* conference, not a theory conference.**
  Favor practical/empirical advice over theoretical novelty. Reviewers
  want clear deployment story + ablations on standard benchmarks.

That's the project. The CSV is the work plan. Help me execute the
nearest todo row.
