# MultiPano — map-anchored multi-panorama → real-geometry 3DGS

A second attempt at the world-models problem, fixing the core weakness of
`3DGS/` (single-panorama → hallucinated parallax).

---

## The problem with iteration 1

Single panorama → monocular depth → one Gaussian per pixel.
Everything lives on a thin spherical shell around the original camera.
Move the viewer sideways and you see (a) "depth skirts" stretched
from silhouette edges, and (b) **black voids** behind every foreground
object — because no pixel ever saw what was behind it.

This is geometry's fault, not the network's. One image cannot encode
parallax.

## The idea

Generate **several panoramas from different positions in the same
imagined scene**, then reconstruct properly with multi-view 3DGS.
With ≥2 panoramas separated by a real baseline, parallax is no longer
hallucinated — it's *measurable*. The occluded regions of panorama A
are visible in panorama B.

## Pipeline (left-to-right)

```
   1. MAP                  2. PANORAMAS                     3. 3DGS SCENE
   ─────                   ───────────                      ────────────
top-down layout        N panoramas, one per           multi-view 3DGS
of the scene,    ──►   marked map position,    ──►    optimization with
N marked camera        each conditioned on map        map-given poses;
positions              + previous panoramas           diffusion regularizer
                       for consistency                fills residual gaps
```

### Stage 1 — the map (first)

A simple top-down illustration of the scene the user wants. Doesn't
need to be photorealistic — it's a layout document. It encodes:

- where each object lives (trees, buildings, paths, characters)
- where each camera will stand (N numbered pins)
- approximate scale (so we can convert map distance → world metres)

The map is generated **once** and frozen. Everything downstream cites it.

### Stage 2 — the panoramas (then, in order)

- **Panorama 1** at map position P1. Generated from a prompt that
  includes (a) the map image, (b) "you are standing at point 1 facing
  X°", (c) the scene's style description. This panorama fixes the
  *visual identity* of the world: lighting, palette, character design,
  building materials, sky.
- **Panorama k (k ≥ 2)** at map position Pk. Generated from a prompt
  including (a) the map, (b) **panorama 1 as image conditioning**, (c)
  "you are standing at point k facing X°". The image conditioning is
  the critical part — text-to-image alone drifts; image-to-image keeps
  the church looking like the same church.

Tools that support image conditioning on a 360° panorama prompt: OpenAI
`gpt-image-1` (image input), Stable Diffusion XL + IP-Adapter,
PanoFusion, MVDiffusion. The first is the simplest to start with.

### Stage 3 — multi-view 3DGS

- **Skip COLMAP.** We don't need to recover camera poses from feature
  matches — we already know them from the map. Hand the poses to the
  optimizer directly.
- **Equirectangular camera model** in the rasterizer, or slice each
  panorama into ~20 perspective views per panorama and run standard
  perspective 3DGS.
- **Loss:** photometric reconstruction against the N panoramas + a 2D
  diffusion regularizer (SDS) on novel views to reconcile any residual
  inconsistencies the generator left behind.

## The main risk

**Cross-view consistency.** Today's image generators don't have an
internal 3D model of the world. Even with the map + image conditioning,
panorama 2 may give you a slightly different church, the trees might
shift, characters may teleport. Multi-view stereo lives or dies on
feature correspondence.

Mitigations, in order of effort:

1. **Image conditioning** (above). Most of the benefit.
2. **Aggressive prompt anchoring** — describe key landmarks verbatim
   each time, lock the style ("painted matte, soft light, 2 PM,
   high overcast").
3. **Small baselines first.** Two panoramas 1–2 m apart fail less than
   two panoramas 20 m apart. Start small, then push.
4. **Diffusion regularizer in the 3DGS loss** (Stage 3) — partial
   inconsistencies get smoothed into a plausible average.
5. **If all else fails:** use a real multi-view-consistent generator
   like CAT3D / MVDream / WonderWorld instead of a generic LLM image
   model.

## First concrete experiment

Before building any pipeline, sanity-check the generator:

1. Generate a top-down map of one scene.
2. Generate panorama 1 at P1 with the map as input.
3. Generate panorama 2 at P2 with map + panorama 1 as inputs.
4. Eyeball: do shared objects look like the same objects from a
   different angle?

If yes → wire up multi-view 3DGS.
If no → swap the generator before writing any 3DGS code.

## Related work (don't reinvent)

- **DreamScene360** (Zhou et al., ECCV 2024) — single panorama → 3DGS
  with diffusion. Path B in the `3DGS/` folder.
- **CAT3D** (Google, NeurIPS 2024) — *multi-view-consistent* image
  generation from one image, then 3DGS.
- **WonderJourney / WonderWorld** (Stanford, 2024) — text → coherent 3D
  scene sequences.
- **LucidDreamer** — text-to-3D scene via diffusion + monocular depth.

This experiment is closest in spirit to CAT3D but with a *human-authored
map* in the loop instead of a model-generated multi-view set.

---

## Reusable ChatGPT prompts

Target model: **gpt-image-2** (OpenAI, released April 2026). Key
capabilities the prompts below rely on:

- Accepts **up to 16 reference images** per call — we can attach many
  stills + the map + previous panoramas all at once.
- **Native thinking mode** — reasons over composition, object counts,
  lighting, and constraints *before* rendering. Long, structured
  prompts with explicit constraints work well; no need to be terse.
- **Strong cross-image consistency** — character, material, and scene
  identity transfer across edits. This is what makes panorama-N stay
  consistent with panorama-1.
- **Better spatial reasoning** — "to the left of", "behind",
  "overlapping" are honoured more reliably than in 1.5.
- **1K / 2K / 4K resolution.** Maps at 2048², panoramas at 4096×2048.

Prompt structure used below — keep this order, it parses cleanly:
**Goal → Inputs (labelled) → What to draw → Constraints → Intended use.**

### Step 1 — Generate the map

**Inputs to attach** (label them by index in the prompt — gpt-image-2
will reference each image by its slot):

| Slot       | Image                                                                 | Required? |
|------------|-----------------------------------------------------------------------|-----------|
| Image 1–N  | 4–8 stills of the **same scene** from the anime episode (mix wide establishing shots, mid-shots, close-ups) | ✅ |
| Image N+1  | Rough hand-drawn sketch of the layout (blobs labelled "chapel", "tree", "path") | optional  |
| Image N+2  | One additional style still you want to match closely (palette / linework) | optional  |

**Prompt:**

> **Goal.** Generate a top-down 2D illustrated map of one fictional
> outdoor scene. This map will be used as a layout reference for
> subsequent AI image generation.
>
> **Inputs.**
> - **Images 1–N** are stills of the same physical location from
>   different angles, moments, and characters. Treat them as multiple
>   views of one place — infer where each feature (building, tree,
>   path, wall, water, prop) sits relative to the others.
> - **Image N+1** (if attached) is a rough hand sketch of the layout.
>   When present, its proportions and relative positions **override**
>   anything you'd infer from the stills.
> - **Image N+2** (if attached) is a dedicated style anchor — copy its
>   linework, palette, and texture closely.
>
> **What to draw.**
> 1. A **top-down floor-plan-style map** of the scene. Every
>    significant feature visible in any input still must appear
>    somewhere on the map.
> 2. **Style** matches the input stills — illustrative watercolor / ink,
>    soft palette, Witch Hat Atelier–like.
> 3. **Composition** — **strict orthographic top-down view**, looking
>    straight down. **Zero perspective skew.** Every object drawn as
>    if seen from directly above; no 3/4, no isometric, no axonometric
>    tilt.
> 4. **Aspect ratio 1:1 (square), 2048 × 2048 px.**
>
> **Do NOT include.**
> - No camera pins, numbered markers, dots, or icons.
> - No grid lines, scale bars, compass roses, or coordinate labels.
> - No text, captions, watermarks, or annotations.
> - No characters or living figures — environment only.
> - No 3/4 / isometric / tilted view. Strict top-down only.
>
> **Intended use.** Layout reference for a multi-view 3D pipeline.
> Pins are added programmatically afterwards. Treat the map as a
> precise spatial document, not a vibe sketch.

### Step 2 — Generate each panorama

**Inputs to attach:**

| Slot     | Image                                                                                | Required?     |
|----------|--------------------------------------------------------------------------------------|---------------|
| Image 1  | The generated map, with the **target camera position circled** (or attach the map with a red dot painted at the camera position) | ✅ |
| Image 2  | One or two style stills (the same anime frames as the map's style reference)         | ✅ |
| Image 3  | **Panorama 1** of this scene — for panoramas with k ≥ 2 only                         | ✅ if k ≥ 2  |

**Prompt:**

> **360 equirectangular panorama** of one fictional outdoor scene,
> viewed from a specific position marked on the attached map.
>
> **Inputs.**
> - **Image 1** is the top-down map of the scene. A red mark / circle
>   shows the **camera position** for this panorama. Specifically: the
>   camera stands at [VIEWPOINT — e.g. "the south-west point near the
>   stone path, ~5 m from the chapel"].
> - **Image 2** is the style anchor. Match its illustrative
>   watercolor / ink style, palette, and linework.
> - **Image 3** (if attached) is panorama 1 of this same scene. **The
>   chapel, trees, walls, sky, time of day, lighting direction, palette,
>   and weather must be identical to Image 3.** Only the vantage point
>   changes. Do not invent new buildings. Do not rearrange large
>   features. Do not add or remove characters.
>
> **What to draw.**
> 1. A **full 360° view** from the marked camera position.
> 2. **Equirectangular projection** (cylindrical equidistant): the sky
>    stretches across the entire top row of the image, the ground
>    across the entire bottom row, vertical objects bow into gentle
>    vertical arcs near the top and bottom.
> 3. **Seamless horizontal wrap** — the leftmost column of pixels must
>    continue into the rightmost column with no visible seam, as if
>    the image were wrapped around a sphere.
> 4. **Yaw convention.** The **centre column of the image faces
>    map-north** (top of the map). The left half of the image is what
>    is west of the camera; the right half is what is east; the seam
>    at the left/right edge is what is directly south.
> 5. **Aspect ratio exactly 2:1. Resolution 4096 × 2048 px.**
>
> **Do NOT include.**
> - No HUD, watermark, logo, or text.
> - No characters not present in Image 3 (preserve cast).
> - No new buildings, paths, or large features not visible on Image 1.
> - No time-of-day, lighting, or weather change from Image 3.
> - No fisheye / wide-angle distortion *other than* what equirectangular
>   projection naturally requires.
>
> **Intended use.** This panorama will be combined with others from
> different positions on the same map into a multi-view 3D
> reconstruction. **Geometric and identity consistency with Image 3
> matters more than artistic novelty.** When in doubt, copy from Image 3
> rather than invent.

### Step 2b — Iterating on a panorama

If panorama-k disagrees with panorama-1 on something specific (the
chapel roof colour drifted, an extra tree appeared, etc.):

> **Change only:** [one specific edit, e.g. "the chapel roof should be
> dark slate grey, matching Image 3"].
>
> **Keep everything else the same** — same composition, same
> projection, same camera position, same lighting, same palette, same
> characters, same buildings, same trees. Re-attach the same inputs as
> before (Image 1: map with the circle, Image 2: style anchor, Image 3:
> panorama 1). Do not introduce any other changes.

### Step 3 — Pipeline explainer image (the diagram)

Used once, for slides / docs. Not part of the runtime pipeline.

> A clean, modern technical infographic showing a 3-stage pipeline laid
> out **left to right** on a soft off-white background, paper-figure /
> SIGGRAPH-explainer aesthetic. 16:9 aspect ratio.
>
> **Stage 1 — "Map":** a stylized top-down illustration of a small
> outdoor fantasy scene (clearing with trees, stone path, small chapel,
> low wall). Three small numbered camera-icon pins labelled **1**,
> **2**, **3** at different points on the map.
>
> **Stage 2 — "Panoramas":** three horizontal equirectangular 360°
> panorama strips at 2:1, stacked vertically and labelled **1**, **2**,
> **3** to match the map pins. Each strip shows the same chapel, same
> trees, same midday light, but from a different vantage point so the
> chapel sits at a different position in each strip.
>
> **Stage 3 — "3D scene":** an isometric or 3/4-perspective view of the
> reconstructed scene rendered as a soft cloud of overlapping translucent
> coloured points (visible Gaussian-splat fuzz throughout, not just at
> silhouettes), with a faint dotted curved arrow tracing a camera
> fly-through path winding past the chapel.
>
> Connecting elements: thin arrows between the three stages (Map →
> Panoramas → 3D scene). Subtle dotted lines link each numbered pin on
> the map to its matching panorama strip.
>
> Style: warm, technical, clean — minimal text (only stage titles and
> small 1/2/3 numerals), soft shadows, restrained colour palette. No
> screenshots, no UI chrome, no extra explanatory text on the image.

---

## TODOs / future work

- **`sketch.html` — in-browser sketcher.** A single HTML file with a
  canvas: drag to draw blobs / lines, label each blob (chapel / tree /
  path / wall) from a small palette, then click *Download PNG* to save
  the sketch as input for Step 1. Mobile-friendly so you can sketch on
  a phone or tablet. Single-file, no build step, drop into `MultiPano/`
  alongside this README.
- **`place_pins.py` — click-to-place pin coordinates.** Open the
  generated map, click N positions, save them as `poses.json`
  (`{px, py, yaw_deg}` per pin). Converts to metric via a declared
  `scale_m_per_px`.
- **`generate_panoramas.py`** — `openai` SDK call that, given the map +
  `poses.json` + Step 2 prompt template, loops over pins and produces
  `pano_1.png`, `pano_2.png`, …  Re-uses panorama 1 as a reference for
  panoramas k ≥ 2 automatically.
- **`recon_3dgs.py`** — multi-view 3DGS optimization with `poses.json`
  as ground-truth poses, equirectangular camera model, optional SDS
  loss. Outputs `scene.ply`.
