# PhysTalk (reimpl) — contributor notes

## Role
I (Claude, contributor) am building a TRAIN-FREE, agent-authored physics system on
EXPLICIT 3DGS objects. Goal: do what surface-smearing physics CANNOT — **shatter,
reveal interior, change material** — on real reconstructed objects. The "PhysTalk core"
is an LLM/agent that WRITES a transform program (composing the op-library) from a text prompt.

## The research wall (why this matters)
Existing physics-on-3DGS (PhysGaussian / Fracture-GS / PhysTalk) mostly TRANSPORT the
visible gaussians. So crushing an object makes it DEFLATE (smear the shell), not shatter +
reveal a hollow/again-filled interior + change material. Attacking that wall is the point.

## Locked constraints (from the user, do not violate)
explicit-3D (NOT video diffusion / frame-gen) · no validation/benchmark framing ·
no robotics · no anime · constructive capability (build, don't measure).

## Build plan (cycles)
1. [DONE c1] `shatter` — k-means fragments + outward velocity + gravity. Beats smear.
2. [DONE c2] interior fill — axis-stabbing solidify (flood-fill FAILED on thin TripoSplat shells); 26k interior gaussians; cuts reveal a solid cross-section.
3. [DONE c3] agent-authoring loop — LLM/agent writes the transform program from a prompt (the PhysTalk core).
4. [DONE c4] more ops — drop+ground-collision, slice/cut along a plane, topple, bend, explode, melt.
5. [DONE c5] gallery — several objects x prompts, as strips/gifs.
6. [DONE c6] writeup — what works / where the wall is / next steps.

## Cycle log
- **C1 (06-30):** built op-library + render pipeline on explicit 3DGS; added `shatter`.
  `c1_shatter.png` shows crush=deflate (the wall) vs shatter=fragments fly apart (fix).
  Observed next wall: fragments are HOLLOW SHELLS -> interior fill is C2.

- **C2 (06-30):** interior-fill. flood-fill-enclosed FAILED (120 voxels) -> TripoSplat objects are thin NON-WATERTIGHT shells (openings leak). Switched to AXIS-STABBING solid test (voxel is inside if surface exists on both sides along all 3 axes) -> 26,878 interior gaussians. `c2_interior.png`: sliced hollow shell = empty cavity; solidified = solid cross-section. Interior is uniform dark shade (crude; heterogeneous/textured interior is a later upgrade). Added `slice_cut` op.

- **C3 (06-30):** AGENT-AUTHORING CORE. Spawned an LLM agent given the op-library signatures + two prompts ("smash into pieces", "cut in half to see inside"); it WROTE `author_smash` (solidify->shatter) and `author_slice` (solidify->slice), correctly ordering solidify FIRST so interiors show. `c3_authored.png`: smash now yields SOLID fragments (interior-fill + shatter compose); slice reveals a solid cross-section. The train-free, agent-authored physics core works end-to-end on a real 3DGS object.

- **C4 (06-30):** textured interior — interior gaussians coloured by NEAREST-SURFACE colour x0.62 + depth-darkening, so a cut shows marbled solid material (vs C2 flat dark). Added ops: `drop` (gravity freefall + impact squash), `explode` (radial burst -> dust cloud), `melt` (slump + lateral spread/pool). `c4_ops.png`. op-library = 7 transforms. Minor: drop falls out of frame (cosmetic; add ground plane / camera-track later).

- **C5 (06-30):** gallery — 6 DISTINCT objects, each a different agent-authored op: ceramic_vase=shatter, dslr_camera=slice, sneaker_shoe=melt, table_lamp=explode, acoustic_guitar=slice, ceramic_teapot=shatter. `c5_gallery.png`: shatter -> solid chunks, slice -> cut reveals interior, explode -> dust burst, melt -> slump. The train-free agent-authored physics GENERALIZES across distinct real reconstructed objects.


## WRITEUP (C6) — morning review

**What this is.** A from-scratch, TRAIN-FREE reimplementation of the PhysTalk/PhysSplat idea:
an LLM agent WRITES a physics transform program (composing a fixed op-library) from a text
prompt, and it runs on an EXPLICIT 3DGS object. Built overnight to *find the wall by building*,
not to be novel yet. ~250 lines, one file (`phystalk.py`), no training, no dataset.

**What works (C1-C5, see c1..c5 .png):**
- `shatter` -> object breaks into SOLID rigid fragments that fly apart (vs surface-smear = deflate).
- `solidify` (axis-stabbing) fills the hollow interior (26k gaussians for the teapot) so cuts/breaks
  reveal a SOLID, TEXTURED cross-section (nearest-surface colour + depth-darkening).
- `slice_cut` parts two halves and shows the interior; `explode` -> dust burst; `melt` -> slump/pool.
- **Agent-authoring core (C3):** an LLM, given only op signatures + a prompt, wrote `solidify->shatter`
  and `solidify->slice` and ordered them correctly. Generalises across 6 distinct objects (C5 gallery).

**The walls we hit (this is the research substance):**
1. TripoSplat objects are THIN, NON-WATERTIGHT shells -> naive flood-fill interior fails (found 120 voxels);
   axis-stabbing solid test fixes it (but overfills concavities slightly).
2. **The interior is FABRICATED, not grounded.** We colour it by the nearest *surface* gaussian — it is a
   plausible guess, NOT the real inside (an apple's core, a camera's circuit board, a layered material).
   Every cut/break reveals an INVENTED inside. <- this is the honest open problem.
3. Physics is KINEMATIC/scripted (heuristic transforms), not a real solver: no true contact, no conservation,
   no material-specific fracture. Fine for a demo; a real method needs a solver or learned dynamics coupling.
4. Cosmetic: `melt` is subtle; `drop` falls out of frame (needs a ground plane).

**Honest novelty position.** As-is this is a *baseline* equivalent to PhysTalk/PhysSplat, NOT a contribution.
But it surfaced one genuine, defensible wall by building: **the revealed interior is ungrounded.** A paper
about *grounding / generating the TRUE interior structure that a physical transformation exposes* (so a cut
fruit shows a real core, a broken phone real innards) is the wedge — and it connects directly to your earlier
amodal/hidden-completion instincts, now aimed at the physical-transformation angle, on explicit 3DGS.

**Next steps (your call in the morning):**
- (A) Chase the INTERIOR-GROUNDING wall: generate plausible/structured interiors (per-material, from a prior)
  that cuts/breaks reveal -> the one thing this whole space fakes.
- (B) Push the agent-authoring: multi-step, conditional, longer physical narratives from language.
- (C) Polish for a hero demo: stronger melt, ground plane, better interior texture, one combined figure.
- Recommendation: (A) is the most paper-shaped and most *you*.

## Run
`ssh node008`; ttt env (gsplat); `python /scratch/.../world-models/phystalk/phystalk.py`

- **P1 (06-30):** hero figure `hero.png` (3x3: teapot shatter / camera slice->interior / lamp explode) with the thesis in the title. Good top-of-review image.

- **P2 (06-30):** polish — `drop` now lands on a floor (stays in frame) + settle squash; `melt` stronger (0.92 collapse, 2.2x pool). `p2_ops.png`.

- **P3 (06-30):** interior-grounding PROBE — `solidify(structured=True)` gives a radial surface-tinted-shell -> warm-dark-core interior. `interior_v2.png`. HONEST: marginally more material-like (a warmer core on the cut) but STILL A GUESS — a real vase is hollow-walled clay, not a solid warm core. Confirms the wedge: heuristic structure is cosmetic; the real contribution = GROUNDING the interior in actual material/structure, not a radial guess.

**POLISH QUEUE EMPTY.** Prototype + figures (c1-c5, hero, p2_ops, interior_v2) + writeup complete. Awaiting user direction (recommend wedge A: interior-grounding).
