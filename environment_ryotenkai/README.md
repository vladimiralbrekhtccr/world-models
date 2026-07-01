# environment_ryotenkai

Workspace for **Spark** (World Labs' web 3DGS renderer) work tied to the
**RYŌTENKAI** talk: *One-Shot Domain Expansion for Walkable 3D Worlds*
(Math+AI conference, Nazarbayev University, Astana, Sep 9–11 2026).

This README has two parts:
1. **Context** I already gathered — so the agent starts oriented.
2. **`## What I need you to do`** — Vladimir fills this in. Everything below
   that header is the brief; everything above is reference.

---

## Context: what world-models is about

This repo explores **image/prompt → explorable 3D world** (World-Labs /
Marble style). The common output format is a **3D Gaussian Splatting** scene
(`.ply` → `.ksplat`) viewed in the browser with **Spark**. Several backbones
feed that viewer:

- **COLMAP + nerfstudio (splatfacto)** — phone video → splat (highest quality).
- **Lyra 2.0** — single image → diffusion video → gsplat optimization.
- **HunyuanWorld 2.0 (HY-World-2)** — the RYŌTENKAI backbone. Single image →
  panorama → video diffusion → feed-forward 3DGS. "Domain expansion": the
  geometry behind/beside the photo is *hallucinated*, then distilled into an
  explicit Gaussian cloud you can fly through. Central thesis = **compounding
  error**: each stage eats the previous stage's output, so final fidelity is
  bounded by the weakest generative link.
- **TripoSplat** — single object → 3DGS (the current chosen direction).

Live demos: `https://kitan-a.com/3dgs/...` (`/plant/`, `/plant-4k/`, `/lyra/`,
`/hyworld/`, `/` room, `/colmap/`). Full pipeline detail lives in the repo
root `CLAUDE.md` (the "3DGS PLAYBOOK").

---

## Spark cheat-sheet (how this repo uses it)

**Import (CDN importmap):**
```html
<script type="importmap">
{ "imports": {
  "three":            "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.180.0/three.module.js",
  "three/addons/":    "https://esm.sh/three@0.180.0/examples/jsm/",
  "@sparkjsdev/spark":"https://sparkjs.dev/releases/spark/2.0.0/spark.module.js"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
```

**Minimal scene:**
```js
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.01, 1000);
const renderer = new THREE.WebGLRenderer({ antialias:false });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const spark = new SparkRenderer({ renderer });   // optional: { maxStdDev: 2.8 }
scene.add(spark);
const splat = new SplatMesh({ url: 'scene.ksplat' });
scene.add(splat);
```

**Coordinate-frame rotation (critical — depends on producer):**
| Source                       | SH degree | Rotation              |
|------------------------------|-----------|-----------------------|
| splatfacto / nerfstudio (+Z-up) | 2     | `splat.rotation.set(-Math.PI/2,0,0)` (per playbook `-X/2`) |
| Lyra 2.0 (OpenCV Y-down)     | 0         | `splat.rotation.set(Math.PI,0,0)`     |
| HunyuanWorld 2.0 (OpenCV)    | 0         | `splat.rotation.set(Math.PI,0,0)`     |
| TripoSplat                   | 0         | `splat.rotation.set(Math.PI,0,0)` (+ `y=π/2` yaw) |

**PLY → .ksplat** (deploy format), node20 env:
```bash
$HOME/miniconda3/envs/node20/bin/node \
  /scratch/vladimir_albrekht/projects/world-models/MultiPano/ksplat/convert.mjs \
  <in.ply> <out.ksplat> <compression=1> <alphaThresh> <shDeg>
# compression 1 = 16-bit, visually lossless, ~70% smaller, fast (USE THIS).
#   compression 0 does NOT render in Spark.
# shDeg: splatfacto=2, Lyra/HY-World/TripoSplat=0.
```

**Orbit + fly controls** pattern exists in
`_temp/roomviewer/index.html` (orbit default w/ damping; fly = WASD move,
R/F up-down, Q/E roll, drag look). Good reference viewers to copy from:
- `learn/index.html` — multiple inline splat demos
- `_temp/roomviewer/index.html` — orbit/fly toggle, metadata-driven up-align
- `TripoSplat_proj/TripoSplat/static/viewer/viewer.html` — group-based rotation

**Deploy:** source viewers live under `_temp/*/index.html`; copy to
`/var/www/3dgs/<name>/` on foggen (Caddy) → served at `kitan-a.com/3dgs/<name>/`.

---

## What I need you to do

<!-- Vladimir: write the brief here. Leave the context above intact. -->
