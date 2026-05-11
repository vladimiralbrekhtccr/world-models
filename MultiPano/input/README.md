# MultiPano · input/

Per-scene reference material lives here. One sub-folder per scene.

## Convention

```
input/<scene_name>/
├── 01_establishing.png    # 4–8 anime stills of the same location,
├── 02_mid.png             # numbered so you can refer to them by index
├── 03_closeup.png         # when prompting ChatGPT ("see Image 03")
├── 04_alt_angle.png
├── sketch.png             # optional — rough hand layout (Step 1 input)
└── style.png              # optional — dedicated style anchor (Step 1 input)
```

After generation, also drop the outputs in the same folder:

```
input/<scene_name>/
├── map.png                # Step 1 output (top-down)
├── poses.json             # camera pin coordinates (one entry per pano)
├── pano_1.png             # Step 2 output, position 1
├── pano_2.png             # Step 2 output, position 2
└── pano_k.png             # ...
```

## Why files, not chat-only uploads

Putting the references on disk has three benefits:

1. **Reproducibility** — generating the same scene twice uses the same
   reference files; no chasing screenshots through chat history.
2. **Tooling** — `place_pins.py` (planned) and `generate_panoramas.py`
   (planned) both read from this folder.
3. **Versioning** — large image files are gitignored, but the folder
   structure itself is tracked so the convention is documented.

## Naming rule of thumb

Prefix with `NN_` so files sort in the order you want ChatGPT to read
them (its multi-image input respects upload order). Use snake_case for
the scene name (no spaces or special characters).
