# world-models

Exploration folder for **world-model** experiments — turning images / video /
panoramas into explorable 3D scenes.

Each experiment lives in its own subfolder. Current ones:

- [`3DGS/`](3DGS/) — one 360° panorama → 3D Gaussian Splatting `.ply`.
  Two paths: a fast depth-only baseline and a full DreamScene360 pipeline.

Compute target: a single H100 80GB on `node007` (interactive use, GPU 0).

Related: <https://github.com/vladimiralbrekhtccr/360-panorama-viewer> — the
browser 360° viewer that motivated this exploration.
