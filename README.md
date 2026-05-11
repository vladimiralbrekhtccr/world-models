# world-models

Exploration folder for **world-model** experiments — turning images / video /
panoramas into explorable 3D scenes.

Each experiment lives in its own subfolder. Current ones:

- [`3DGS/`](3DGS/) — one 360° panorama → 3D Gaussian Splatting `.ply`.
  Two paths: a fast depth-only baseline and a full DreamScene360 pipeline.
  Live demo: <https://vladimiralbrekhtccr.github.io/360-panorama-viewer/3dgs/>
- [`MultiPano/`](MultiPano/) — concept doc for v2: map-anchored
  multi-panorama generation → multi-view 3DGS (real parallax, no
  single-image hallucination).

Compute target: a single H100 80GB on `node007` (interactive use, GPU 0).

Related: <https://github.com/vladimiralbrekhtccr/360-panorama-viewer> — the
browser 360° viewer that motivated this exploration.
