# Learn 3DGS — text fallback

> **The primary version is now an interactive HTML course** with widgets
> integrated through every module:
> ### 👉 <https://kitan-a.com/3dgs/learn/>
>
> Read that. Come back to this file only if you need a printable / offline
> text version. The HTML course teaches the same material with widgets
> you can poke — far better for a visual topic like this.

A read-top-to-bottom guide to **3D Gaussian Splatting** and the world-model
work in this repo. Written so that after each module you can *answer
questions* about it — not just recognise the words.

How to read this:
- Each module has a plain-language explanation, then **sidebars** (the
  light math — skip on first read, return when curious).
- **❓ You can now answer** — the questions you should be able to field.
- **👥 Tasks you could delegate** — turning the concept into work for
  teammates, so you know what to hand off.

Demos referenced (all live on kitan-a.com):
`plant` = classical optimization · `lyra` = generative · `hyworld` =
feed-forward · `garden` = splat→mesh.

---

## Module 1 — The problem & the big idea

**The problem: novel view synthesis.** You have a handful of photos of a
scene. You want to render that scene from *new* camera positions you never
photographed — to fly through it, orbit it, walk into it. The photos are
flat and finite; the goal is a continuous, explorable 3D representation.

**How people did it before:**

- **Photogrammetry → textured mesh.** Match features across photos,
  triangulate a 3D mesh, paint textures on it. Works well for solid
  opaque objects; struggles badly with thin structures (leaves, wires,
  hair), fuzzy things, transparency, and reflections.
- **NeRF (2020).** Represent the whole scene as *one neural network* that
  answers: "at this 3D point, looking in this direction, what colour and
  how dense?" Render an image by shooting a ray per pixel and sampling the
  network along it. Photorealistic — but originally *minutes* to render a
  single frame, and slow to train. NeRF is an **implicit** representation:
  the scene is a function, hidden inside network weights.

**The big idea of 3DGS (2023).** Instead of a neural network, represent
the scene as **a literal list of millions of tiny 3D blobs** —
*Gaussians*: fuzzy, semi-transparent, coloured ellipsoids. To render,
you just project each one to the screen and blend them. No neural
network runs at render time.

> **The key shift, in plain words: NeRF is a *calculator*. 3DGS is a
> *spreadsheet*.**
> A calculator doesn't store `2+2=4` anywhere — you ask, it *computes*.
> A spreadsheet has `4` already sitting in a cell — you *read* it.
> NeRF computes every pixel by asking a neural network "what's at this
> point in 3D?" — slow, because computing. 3DGS already has the scene
> written down as millions of Gaussians — fast, because just reading +
> a tiny bit of blending.
>
> The fancy words for this are *implicit* (calculator — hidden inside a
> function) and *explicit* (spreadsheet — written down directly). Don't
> let the words intimidate you: calculator vs spreadsheet is the whole
> idea.

The payoff:
- **Real-time** — 100+ FPS rendering, vs NeRF's seconds-to-minutes
  (because you're reading, not computing).
- **Fast training** — minutes to ~an hour, vs hours/days.
- **Works in a browser** — a browser can read a "spreadsheet of blobs"
  and do simple blending. Running a neural-net calculator on every pixel
  is too heavy.
- **Tooling** — an explicit list of blobs can be streamed, edited,
  compressed, and rendered on any laptop. A neural network can't be
  sliced up and shipped the same way.

This also explains two side-facts you'd otherwise have to memorize:
- **3DGS files are big** — a spreadsheet with millions of rows is big.
  (NeRF's network file is small, but you pay for it at render time.)
- **You usually have one file per scene** — every scene gets its own
  spreadsheet.

❓ **You can now answer:**
- What is "novel view synthesis"?
- Why is a textured mesh a poor fit for foliage or a 360° garden?
- What did NeRF do, and what was its core weakness?
- In one sentence, what is the 3DGS idea, and why is it fast?
- What does "explicit vs implicit scene representation" mean?

👥 **Tasks you could delegate:**
- Have someone collect 3–4 reference papers (NeRF, 3DGS, Mip-NeRF 360)
  and write a one-paragraph summary of each — see `READING.md`.

---

## Module 2 — The Gaussian (the primitive)

A **3D Gaussian** is the atom of the scene. Picture a fuzzy, coloured,
semi-transparent ellipsoid: solid-ish in the middle, fading smoothly to
nothing at its edges. A scene is millions of these, overlapping.

Each Gaussian stores a small bundle of numbers:

| Property | Numbers | Meaning |
|---|---|---|
| **Position** (mean) | 3 | where its centre sits in 3D |
| **Scale** | 3 | how long it is along its 3 local axes |
| **Rotation** | 4 (quaternion) | how that ellipsoid is oriented |
| **Opacity** | 1 | how solid vs see-through it is |
| **Colour** | 3 → 48 | its colour, possibly view-dependent (see below) |

Scale + rotation together describe the ellipsoid's **shape and
orientation** — formally its *covariance*. Position + covariance +
opacity + colour is the whole primitive.

**Why "Gaussian"?** The opacity doesn't cut off sharply at the edge — it
falls off following the Gaussian (bell-curve) function: maximum at the
centre, fading smoothly outward. That soft falloff is *the* reason
Gaussians blend into a continuous image instead of looking like a pile of
hard pebbles.

**Why ellipsoids, not spheres? (anisotropy).** A flat wall is best
covered by flat, pancake-shaped Gaussians. A thin twig by one long
skinny Gaussian. If Gaussians were forced to be round spheres, you'd need
vastly more of them to cover the same surfaces. Letting each one stretch
independently along 3 axes — *anisotropy* — is what makes 3DGS efficient.
**Try it:** in the interactive demo, squash one axis to near-zero and
you get the flat "splat" that tiles a surface.

**View-dependent colour (spherical harmonics).** A matte wall looks the
same colour from every angle. A glossy apple has a highlight that *moves*
as you walk around it. To capture that, a Gaussian's colour isn't one RGB
triple — it's a small function of viewing direction, encoded as
**spherical harmonics (SH)** coefficients:
- **SH degree 0** = 3 numbers = one flat colour (no view dependence).
- **SH degree 3** = 48 numbers = colour varies with angle (captures
  highlights, sheen).

So a full Gaussian is ~59 numbers (SH-3) or ~17 numbers (SH-0). *Our
WorldMirror and Lyra outputs are SH-0; splatfacto outputs SH-3.* That is
why the ksplat conversion uses `shDeg 0` for the former and `2` for the
latter.

> **Sidebar — the covariance matrix.** The ellipsoid's shape is a 3×3
> *covariance matrix* Σ. It is built from the scale `S` (a diagonal
> matrix) and rotation `R` as **Σ = R S Sᵀ Rᵀ**. Storing `S` and `R`
> separately (instead of Σ directly) guarantees Σ stays a valid
> ("positive semi-definite") covariance during optimisation — you can't
> accidentally produce an impossible ellipsoid. The Gaussian's density at
> a point `x` is `exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ))` — 1 at the centre `μ`,
> decaying with distance, stretched by Σ.

❓ **You can now answer:**
- What numbers define a single Gaussian?
- Why are they called "Gaussians" — what does the bell curve do for us?
- What is anisotropy and why does it make 3DGS efficient?
- What are spherical harmonics for? What's the difference between SH
  degree 0 and 3?
- Why do our different pipelines need different `shDeg` at export?

👥 **Tasks you could delegate:**
- Ask someone to open a `.ply` in a text/hex viewer and list its
  properties (`x, y, z, scale_*, rot_*, opacity, f_dc_*, f_rest_*`) —
  confirming a real file matches this table.

---

## Module 3 — Building a scene: optimization

How does a folder of photos become millions of fitted Gaussians? Classical
3DGS does it by **per-scene optimization** — every scene is its own little
training run.

**Step 1 — camera poses (COLMAP / Structure-from-Motion).** Before any
Gaussians, you must know *where each photo was taken*. COLMAP does this:
it detects features, matches them across photos, and solves for every
camera's position + orientation, plus a **sparse 3D point cloud** as a
by-product. (You have the COLMAP infographic — that whole poster is just
this one step.)

**Step 2 — initialize.** Put one Gaussian at each sparse COLMAP point.
Maybe ~100k Gaussians, rough, blobby, wrong colours. This is the seed.

**Step 3 — the differentiable rasterization loop.** Repeat ~30,000 times:
1. Pick a training photo (whose camera pose you know).
2. **Render** the current Gaussians from that camera → a predicted image.
3. **Compare** predicted vs real photo → a *loss* (how wrong it is).
4. **Backpropagate** — because the renderer is *differentiable*, you can
   compute, for every Gaussian, how to nudge its position / scale /
   rotation / opacity / colour to make the next render less wrong.
5. **Update** every Gaussian by that nudge (gradient descent).

Over thousands of iterations the blobs slide, stretch, recolour, and
sharpen until the renders match the photos from every training angle.

**Step 4 — adaptive density control.** Periodically the optimiser also
*changes how many* Gaussians there are:
- **Clone / split** Gaussians where detail is missing (under-reconstructed).
- **Prune** Gaussians that became nearly transparent or absurdly large.

So the count grows from ~100k to several million, concentrated where the
scene needs detail. Total time: minutes to ~an hour on one GPU.

**The key limitation:** this is **per-scene**. The result is one specific
scene; nothing is reused for the next one. And you need many well-covered
photos — gaps in coverage become holes or smears. (Module 6 is about how
newer methods escape this.)

> **Sidebar — the loss.** 3DGS uses `(1−λ)·L1 + λ·D-SSIM` — L1 is
> average per-pixel colour error; D-SSIM is a structural/perceptual term
> that cares about local patterns, not just raw pixel values. The mix
> (λ≈0.2) gives both colour accuracy and crispness.
> **Sidebar — "differentiable" is the whole trick.** A normal renderer is
> a one-way street: scene → image. A *differentiable* renderer also runs
> backward: given "the image was wrong *here*", it tells you which
> Gaussian parameters to blame and by how much. That backward signal
> (the gradient) is what makes optimization possible at all.

**Tie to a demo:** `plant` (splatfacto-big) is exactly this pipeline —
phone video → frames → COLMAP → 30k-iteration optimization → splat. See
`CLAUDE.md` section A and the recording recipe in section B.

❓ **You can now answer:**
- What does COLMAP do, and why must it run *before* 3DGS training?
- Walk through one iteration of the optimization loop.
- What does "differentiable rasterization" mean and why is it essential?
- What is adaptive density control (clone/split/prune) for?
- Why is classical 3DGS called "per-scene", and why is that a limitation?

👥 **Tasks you could delegate:**
- Have a teammate film a new scene following the recording recipe
  (`CLAUDE.md` §B) and run `run_nerfstudio_big.sh` end-to-end.
- Ask someone to compare `splatfacto` vs `splatfacto-big` on the same
  capture and report the gaussian-count / sharpness / file-size trade.

---

## Module 4 — Rendering in the browser

Rendering 3DGS is called **splatting**, and it is deliberately simple —
that simplicity is why it runs in a browser at 60+ FPS.

For every frame, for every Gaussian:
1. **Project** its 3D ellipsoid onto the 2D screen. A 3D ellipsoid
   projects to a 2D ellipse — a "splat".
2. **Sort** all the splats by depth (far → near).
3. **Blend** them front-to-back: walking from nearest to farthest, each
   splat contributes `colour × opacity × its 2D Gaussian falloff`, and
   you accumulate until the pixel is opaque.

That's it: **project, sort, blend.** No neural network, no ray-marching.
This is *rasterization* — the kind of work GPUs were built for — which is
why it's real-time where NeRF's ray-marching was not.

**Why it works in a browser.** WebGL / WebGPU handle the projection and
blending natively. The only fiddly part is the depth **sort** — it has to
be redone whenever the camera moves — and modern viewers do it fast
enough (GPU sorting / approximate sorting).

**Viewers** (the JS libraries that do the above):
- **antimatter15's** original web splat viewer — the first one.
- **mkkellogg `GaussianSplats3D`** — popular three.js integration.
- **Spark** (by World Labs, `sparkjs.dev`) — what *our* demos use.

**File formats** (how the Gaussian list is stored on disk):

| Format | What it is | Notes |
|---|---|---|
| `.ply` | raw training output | big; ASCII or binary; every property explicit |
| `.splat` | antimatter15's compact binary | 32 bytes/Gaussian; the `garden.splat` we used |
| `.ksplat` | mkkellogg's chunked format | compression levels; `1` = 16-bit, ~65% smaller, visually lossless |
| `.spz` | Niantic's compressed format | gzip-friendly, good for the web |

**Coordinate conventions — a real gotcha.** Different tools assume
different "up" axes. Splatfacto PLYs are +Z-up; Lyra and WorldMirror are
OpenCV-style +Y-down. In the viewer you fix this by rotating the splat
mesh (`X = π` for the Y-down ones). Get it wrong and the scene loads
upside-down.

**Tie to a demo:** every kitan-a.com page. `/3dgs/plant/`, `/3dgs/lyra/`,
`/3dgs/hyworld/` all load a `.ksplat` into a Spark viewer with orbit + fly
controls.

> **Sidebar — splatting vs ray-tracing.** Ray-tracing asks, per pixel,
> "what do I hit?" — you march along a ray. Splatting asks, per
> primitive, "where do I land?" — you project it onto pixels. For
> millions of small fuzzy primitives, projecting + blending is far
> cheaper than marching millions of rays, and it maps perfectly onto
> rasterization GPU hardware.

❓ **You can now answer:**
- What are the three steps of rendering a splat? ("project, sort, blend")
- Why is splatting real-time when NeRF rendering wasn't?
- Which step has to be redone every time the camera moves, and why?
- Name the file formats and when you'd use `.ksplat` vs `.ply`.
- Why do some scenes load upside-down, and how is it fixed?

👥 **Tasks you could delegate:**
- Have someone build a new viewer page for a scene (clone an existing
  `index.html`, swap the asset, set the rotation).
- Ask a teammate to benchmark the same scene as `.ply` vs `.ksplat`
  (load time, file size, visual diff).

---

## Module 5 — Splat → mesh

A splat is wonderful to *look at* but it is **not a surface**. It's a
cloud of overlapping translucent blobs — no faces, no edges, no "skin".
Sometimes you need an actual polygon **mesh**.

**Why convert to a mesh:**
- Game engines (Unity, Unreal) and 3D tools (Blender) expect meshes.
- Physics, collision, and shadows need real surfaces.
- A mesh is much smaller and renders on *any* hardware — no special
  splat renderer required.

**The problem:** you have to *extract* a surface from blobs that were
never a surface. Main approaches:

| Method | Idea | Trade-off |
|---|---|---|
| **Poisson reconstruction** | treat Gaussian centres as an oriented point cloud, fit one watertight surface | smooth; over-blurs fine detail |
| **Marching cubes** | sample the Gaussians' combined density on a 3D grid, extract the isosurface | more faithful; needs a fine grid |
| **TSDF fusion** | render depth maps from many views, fuse into a volume | good geometry; needs the render step |
| **SuGaR / 2DGS** | train the splat to be *surface-like in the first place* | best meshes; must retrain the splat |

**The honest catch:** it is **lossy**. Splats can represent fuzzy things
— foliage, hair, grass — that genuinely *aren't* surfaces. A mesh of a
garden will always be blobbier than the splat. The fix is to mesh only
what really is a surface (filter + crop) and accept the rest won't
convert well.

**Tie to a demo:** `garden` — see the `garden-mesh-viewer` repo. The
recipe: parse `.splat` → **filter** (drop low-opacity floaters, drop huge
background Gaussians) → **crop** to the table region → estimate normals →
Poisson depth 11 → keep the largest connected component → transfer
vertex colours. The lesson we learned: *naive Poisson on the raw splat is
a blobby mess; filtering and cropping before reconstruction is the whole
game.*

❓ **You can now answer:**
- Why isn't a splat already a mesh?
- Give three reasons you'd want a mesh instead of a splat.
- Name two splat→mesh methods and their trade-offs.
- Why is splat→mesh fundamentally lossy?
- What did filtering/cropping do for the garden mesh?

👥 **Tasks you could delegate:**
- Have someone tune `splat_to_mesh.py` params (bbox, opacity cutoff,
  Poisson depth) on a new scene and compare results.
- Ask a teammate to try a marching-cubes or SuGaR path and benchmark it
  against our Poisson output.

---

## Module 6 — Beyond per-scene optimization

Modules 3–4 describe **classical 3DGS**: per-scene optimization, needing
many photos and ~30 minutes each. Two newer families break out of that.

**1. Feed-forward reconstruction.** Train one big neural network on
*thousands* of scenes. At test time, give it a few images and it
**predicts the Gaussians directly in a single forward pass** — no
per-scene optimization. About **one second** instead of 30 minutes.
- Lineage: DUSt3R → MASt3R → VGGT → **WorldMirror**.
- It also predicts the camera poses itself — *no COLMAP needed*.
- Trade-off: slightly softer than a fully-optimized splat, because it
  generalizes from training data instead of fitting your exact scene.
- **Tie to a demo:** `hyworld` — HunyuanWorld's WorldMirror 2.0. We swept
  14 scenes; each ran in ~1 second on one GPU at 2.35 GB VRAM.

**2. Generative.** Use a **diffusion model** to *imagine* a scene — or the
video of moving through it — from a text prompt or a single image,
including the parts that were never photographed.
- **Lyra 2.0** — single image → diffusion generates an orbit video →
  gsplat optimization turns that video into a clean splat. The diffusion
  *invents* the unseen sides. **Tie to a demo:** `lyra` (the chapel).
- **HunyuanWorld 2.0** — a full 5-stage pipeline: text/image → 360°
  panorama → camera trajectories → photoreal video → mesh/splat. See
  `HUNYUANWORLD_PIPELINE.md` for the stage-by-stage breakdown.

**The core trade-off — write this on a slide:**

| | Optimization (classical) | Feed-forward | Generative |
|---|---|---|---|
| Input | many photos | a few images | text or 1 image |
| Speed | ~30 min/scene | ~1 second | seconds–minutes |
| Faithful? | yes — measured | mostly | partly — it *hallucinates* |
| Unseen areas | holes | weak | invented plausibly |

❓ **You can now answer:**
- What does "feed-forward" mean, and how is it different from per-scene
  optimization?
- Why does WorldMirror not need COLMAP?
- What does a generative method add that optimization cannot?
- Where does Lyra use diffusion, and where does it use optimization?
- Classify each of our four demos as optimization / feed-forward /
  generative.

👥 **Tasks you could delegate:**
- Have someone run WorldMirror on a fresh set of images (`run_worldmirror.sh`)
  and report inference time + gaussian count.
- Ask a teammate to read `HUNYUANWORLD_PIPELINE.md` and explain back the
  five stages — a good comprehension check for the whole topic.

---

## Module 7 — The landscape & where we fit

**The goal of this project:** "image or prompt → explorable 3D world."
The same goal companies like **World Labs** (Fei-Fei Li's company) and
**Marble** are chasing. 3DGS is the rendering substrate that makes it
practical — fast, explicit, web-deployable.

**Our four demos map cleanly onto the three families:**

| Demo | Family | What it proves |
|---|---|---|
| `plant` | classical optimization | faithful capture of a real object from video |
| `garden` (mesh) | splat → mesh | the path to game-engine / standard 3D assets |
| `hyworld` | feed-forward | a real scene reconstructed in ~1 second |
| `lyra` | generative | a *navigable* scene hallucinated from one image |

**What's still hard (so you can speak to the open problems):**
- **Unbounded scenes** — skies and far backgrounds have no clean surface.
- **Dynamic scenes** — moving content needs 4D Gaussians (4DGS), an
  active research area.
- **Relighting / editing** — a baked splat has lighting frozen in; making
  it editable is unsolved-ish.
- **Multi-GPU cost** — the full generative pipelines (HunyuanWorld's
  upper stages) still need clusters, not one GPU.

### Glossary

- **3DGS** — 3D Gaussian Splatting; a scene as millions of Gaussians.
- **Gaussian** — a fuzzy coloured ellipsoid; the scene primitive.
- **NeRF** — earlier neural, implicit scene representation.
- **Novel view synthesis** — rendering a scene from un-photographed angles.
- **COLMAP / SfM** — recovers camera poses + sparse points from photos.
- **Rasterization / splatting** — project-sort-blend rendering.
- **Differentiable rendering** — a renderer that also runs backward, to
  give gradients for optimization.
- **Adaptive density control** — clone/split/prune during training.
- **Spherical harmonics (SH)** — encodes view-dependent colour.
- **Anisotropy** — Gaussians stretching unequally along their 3 axes.
- **Feed-forward reconstruction** — predict the splat in one network pass.
- **Diffusion model** — generative model; here, imagines unseen views.
- **Poisson reconstruction** — fits a watertight mesh to oriented points.
- **`.ply` / `.splat` / `.ksplat` / `.spz`** — splat file formats.

### Where to go next

- `READING.md` — the paper list (start with the 3DGS 2023 paper).
- `HUNYUANWORLD_PIPELINE.md` — the full generative pipeline.
- `CLAUDE.md` — the hands-on playbook for every pipeline in this repo.
- The interactive demo — <https://kitan-a.com/3dgs/learn/>.

❓ **You can now answer (the whole-topic check):**
- What is 3DGS, in two sentences, to a non-expert?
- How does a pile of photos become a splat?
- Why does it render in a browser when NeRF couldn't?
- How — and why — do you turn a splat into a mesh?
- What is HunyuanWorld 2.0, and how does it differ from classical 3DGS?
- Which of the three families does each of our demos belong to?

👥 **Now you can scope teammate work:** capture & optimization
(Module 3), viewer/front-end (Module 4), mesh extraction (Module 5),
testing feed-forward & generative models (Module 6). Each module's
delegate-box is a starter task list.
