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

Fill in the `[BRACKETED]` slots. For each generation, attach the relevant
inputs as image inputs (the model needs to *see* the references, not just
read them).

### Step 1 — Generate the map

Inputs to attach: 2–4 anime stills for style + your rough layout sketch
(can be a hand drawing or even a text-described block diagram).

> Generate a **top-down 2D map illustration** of [SCENE — e.g. "a small
> forest clearing with a stone chapel near the centre, a winding stone
> path entering from the south, three tall pine trees on the west side,
> and a low moss-covered wall along the north edge"].
>
> Style reference: match the attached anime stills (illustrative
> watercolor / ink, soft palette, Witch Hat Atelier–like).
> Layout reference: the attached rough sketch defines relative positions
> — preserve them.
>
> Strict requirements:
> - **Strict orthographic top-down view** (looking straight down, zero
>   perspective skew — every object drawn as if seen from directly above).
> - **Aspect ratio 1:1 (square), 2048 × 2048 px.**
> - **No camera pins, no numbered markers, no grid lines, no scale bars,
>   no text annotations.** Pure illustrated map only — we'll add pins
>   programmatically afterwards.
> - Show the relative positions of every object clearly. Spatial layout
>   matters more than artistic flourish.

### Step 2 — Generate each panorama

Inputs to attach: the same anime stills (style), the generated map (with
the pin position you're currently shooting from circled or otherwise
indicated), and *for panorama 2+*, panorama 1 (so the model preserves
identity of the same scene).

> Make a **full 360° equirectangular panorama photo** of [SCENE +
> VIEWPOINT — e.g. "the same forest clearing as in the attached map,
> shot from position **2** marked on the map (south-west of the chapel,
> on the stone path). The chapel should appear in the northern half of
> the panorama; the pine trees are on my right; the moss-covered wall
> is in the far distance behind me."].
>
> Style reference: match the attached anime stills.
> Scene identity reference: match the chapel, trees, lighting, and
> palette of the attached panorama 1 — same world, different vantage
> point. Do not invent new buildings or rearrange large features.
>
> Strict requirements:
> - **Aspect ratio exactly 2:1** (e.g. 2048 × 1024).
> - **Equirectangular projection**: sky stretches across the entire top,
>   ground across the entire bottom, vertical objects bow into gentle
>   vertical arcs. The image must **wrap seamlessly left-to-right**
>   (leftmost column continues into rightmost column).
> - Convention: the **centre column of the image faces map-north** for
>   every panorama (locks yaw so we don't have to think about it).

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
