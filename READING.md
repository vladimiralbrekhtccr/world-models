# Reading list — explorable 3D world generation

Goal: build something like **World Labs / Marble** — text / image / panorama
/ video → an explorable 3D Gaussian-splat world. World Labs is closed; the
list below is the open literature that covers the same ideas. Read top to
bottom.

## Start here — closest to the World-Labs concept

- [ ] **HunyuanWorld 1.0** — *Generating Immersive, Explorable, Interactive
  3D Worlds from Words or Pixels* (Tencent, 2025)
  <https://arxiv.org/abs/2507.21809>
  The most World-Labs-like paper that is actually open-source. Basically
  "open Marble". Read first.

- [ ] **WonderWorld** — *Interactive 3D Scene Generation from a Single
  Image* (Stanford, 2024) <https://arxiv.org/abs/2406.09394>
  Single image → connected explorable scenes, <10 s each. The
  interactive-expansion idea (walk to an edge, generate more world).

- [ ] **Bolt3D** — *Generating 3D Scenes in Seconds* (Google, 2025)
  <https://arxiv.org/abs/2503.14445>
  Feed-forward latent diffusion → full 360° 3DGS scene in <7 s. No
  per-scene optimization, no COLMAP.

- [ ] **WorldExplorer** — *Towards Generating Fully Navigable 3D Scenes*
  (TUM, 2025) <https://arxiv.org/abs/2506.01799>
  Focus on full navigability — no holes/ghosting when you leave the
  capture path. Exactly the failure mode we hit with Lyra.

## Foundations — the building blocks

- [ ] **CAT3D** — *Create Anything in 3D with Multi-View Diffusion*
  (Google, 2024) <https://arxiv.org/abs/2405.10314>
  Multi-view diffusion as the 3D prior. Both Bolt3D and WonderWorld
  build on it.

- [ ] **Beyond Pixel Histories: World Models with Persistent 3D State**
  (2026) <https://arxiv.org/abs/2603.03482>
  Keeping the world persistent/consistent across a long exploration,
  not just per-frame pretty.

- [ ] **3D Gaussian Splatting for Real-Time Radiance Field Rendering**
  (Kerbl et al., SIGGRAPH 2023) <https://arxiv.org/abs/2308.04079>
  The representation everything above outputs. The original 3DGS paper.

- [ ] **Lyra** (NVIDIA) — video-diffusion → 3DGS, feed-forward.
  Already installed on the cluster at `/scratch/.../lyra/`. Re-read
  knowing how the others frame the problem.

## Tools / infra

- **Spark** — World Labs' open-source 3DGS web renderer (three.js / WebGL2,
  streaming LoD). <https://www.worldlabs.ai/blog/spark-2.0> — candidate to
  replace the mkkellogg viewer on the kitan-a.com/3dgs pages.
- **awesome-3d-diffusion** — living paper list to track the field.
  <https://github.com/cwchenwang/awesome-3d-diffusion>

## Takeaway for our pipeline choice

- Classical path (what we run now): COLMAP → splatfacto, per-scene
  optimization. Needs COLMAP, ~30 min/scene.
- Diffusion / feed-forward path (the World-Labs direction): a network
  predicts gaussians directly. **No COLMAP, no per-scene optimization.**
- Likely base to build on: **HunyuanWorld 1.0** (open, image/text input,
  explorable output).
