# Paper plan — VLM-authored, cut-anywhere 3DGS interiors

_Living doc. Updated 2026-07-01. Source of truth for the logic of this project._

## 0. One-line claim
Objects reconstructed as 3D Gaussian splats are **hollow shells**; we give them a
**believable interior** — authored by a **VLM's world knowledge** and made
**consistent from any cut direction** by **differentiable multi-cut optimization** —
with **no interior ground truth** and **no per-cut generation**. (The interior is the
*ultimate* hidden-region completion — fully occluded — connecting to the out-of-frame /
amodal line of work.)

## 1. The method (what we built)
```
whole-object photo
  -> TripoSplat            : image -> 3DGS exterior SHELL (hollow)
  -> VLM (real, per-object): reads the photo, emits a JSON interior SCHEMA
                             (layers-by-depth + inclusions: seeds/core/pit)
  -> procedural fill       : gapless solid (2-of-3 axis enclosure; flood-fill leaks on
                             non-watertight shells) + native anisotropic gaussians;
                             colour by schema, seeds scale with fill density
  -> differentiable fit    : gsplat is differentiable -> optimise the interior gaussians
                             (colour/scale/opacity; surface frozen) over MANY random cut
                             planes vs a perceptual signal:
                               - DINOv2 vs a real cut-photo when one exists
                               - CLIP image-text (reference-less) otherwise
                             optimising over many cuts FORCES 3D coherence -> cut-anywhere
  -> reveal                : cut/break/slice = geometric op on a fully-specified object;
                             any plane -> valid, mutually-consistent cross-section
```
Nothing is trained (frozen TripoSplat + VLM + frozen perceptual/CLIP prior + procedural code).
Two regimes: **instant** (schema fill, seconds, no optimisation) and **metric-fit**
(per-object optimisation, minutes, higher realism).

## 2. Novelty cell & the honest risk
Defensible cell (from the novelty scan, log #47): **{VLM world-knowledge prior} x
{no diffusion-in-the-loop} x {structured, not texture} x {revealed by topology change} x
{cut-anywhere 3D-coherent}**. No prior method occupies all of these.
- **Neighbours:** FruitNinja (per-cut 2D diffusion *texture*, not agentic), GaussianFluent
  (diffusion-densify fracture), PhysTalk (surface only, no interior), TopoGaussian
  (per-object opt, load-bearing not semantic), Articraft (agent-writes-program but meshes).
- **Reviewer risk (real):** "FruitNinja with a VLM swapped in." Beaten ONLY by demonstrating
  the two things it structurally cannot do: **cut-consistency** and **semantic-correct
  structure** (not texture). Novelty lives entirely there — must be *shown*, not asserted.

## 3. The three experiments that make it a paper
1. **Cut-consistency (BACKBONE — our structural win, no GT needed).** Cut the same object N
   ways; measure appearance/feature agreement of overlapping interior regions across cuts.
   We optimise ONE 3D volume -> consistent; FruitNinja inpaints each 2D cut independently ->
   cannot be consistent. This is the cleanest, most honest number and the paper's spine.
2. **Semantic-correctness preference study.** Blind human (+ VLM-judge as scalable secondary,
   randomised, report both directions -- VLM judges have position bias): "which cut looks
   more like the inside of a real <X>?" ours vs FruitNinja vs texture-fill.
3. **Head-to-heads (mandatory).** vs FruitNinja (structure + consistency + runtime); vs
   PhysGaussian/PhysTalk (no interior at all); position vs GaussianFluent (diffusion-in-loop).

## 4. Status: done vs missing
DONE:
- Method end-to-end; 4 objects (watermelon, onion, apple, kiwi); cut-anywhere qualitative.
- Live image TURNTABLE demo (fast, clean) + Spark 3D fallback: kitan-a.com/3dgs/dwm/demo
- Explainer (corrected to real method): kitan-a.com/3dgs/dwm
- Paper draft (Method/Experiments/Limitations + 4-object figure, ~5pp): phystalk/paper/main.tex
- Code committed: github vladimiralbrekhtccr/world-models (phystalk/)
- Research log #48-#62: kitan-a.com/3dgs/research-log-wm

MISSING for a real submission:
- (a) cut-consistency METRIC + numbers  [highest leverage]
- (b) preference study (human + VLM-judge)
- (c) a FruitNinja baseline (their code, or a reimpl; we have texture-fill as a proxy)
- (d) ~20-40 objects at CONSISTENT quality  (apple/onion class must be fixed -- see below)
- (e) ablations: schema vs no-schema; with/without metric-fit; DINO vs CLIP signal

## 5. Quality state (honest, per object)
- STRONG: watermelon (red flesh + seeds), kiwi (green flesh + pale star core + seed ring).
- WEAKER: onion (rings clear but CLIP over-saturates colour), apple (UNIFORM cream interior ->
  CLIP adds smudge artifacts). Uneven quality reads as cherry-picked -> must be closed.
- Fixes queued: SDS prior for uniform interiors (SD v1.5 cached; needs live tuning);
  stronger schema-anchor / better prompts for CLIP; floater-cleaning (done for the turntable).
- Known limitations (in the paper): needs clean single-object-on-plain-bg inputs (cluttered
  photos -> matting floaters); depth-layer schema can't do ANGULAR segments (orange wedges,
  apple star) -> future angular/segment primitive; believable-not-instance (no true-interior GT).

## 6. Venue read (honest)
As-is (4 objects, no quantitative eval) -> strong **workshop / WACV / BMVC** tier.
To credibly target **CVPR/ICCV main**: the 3 experiments + object scale, PLUS one amplifier:
- **Physics/behaviour:** interior isn't just appearance -- a cut/BREAK behaves (soft flesh
  vs hard seed vs a device's parts). Broadens impact; the "world-model" hook.
- **Consistency-as-contribution:** frame "per-cut 2D interior generation is fundamentally
  inconsistent; we introduce the cut-consistency metric + a 3D-coherent method that satisfies
  it," making FruitNinja the baseline our metric exposes.

## 7. MVP path (smallest submittable paper), prioritised
1. **Cut-consistency metric + run it** (ours vs texture-fill vs a per-cut variant) -> win numbers. [DO FIRST]
2. Fix the apple/onion quality class (SDS for uniform; anchor-tune CLIP) -> quality bar.
3. Scale to ~25 objects meeting that bar.
4. Preference study (modest human N + VLM-judge).
5. Write around cut-consistency as the backbone; add ablations.

## 8. Open questions for the user
- Target venue/deadline? (sets scope: WACV-ish MVP vs CVPR with an amplifier)
- Add the physics/behaviour amplifier, or keep it appearance-only?
- Human study logistics (who/how many raters)?
- Is "believable-not-instance" acceptable as the framing, or do we want any grounded-interior
  sub-claim (e.g., transparent/translucent objects where the exterior DOES constrain the inside)?
