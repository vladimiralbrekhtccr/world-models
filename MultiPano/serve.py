#!/usr/bin/env python3
"""Tiny static server for MultiPano/ with annotation + input-gallery persistence.

Run from the MultiPano/ directory:
    python serve.py            # binds 0.0.0.0:8765
    python serve.py 9000       # custom port

Then open http://localhost:8765/  (port-forward through SSH if remote).

Endpoints
---------
    GET  /annotations.json          → MultiPano/annotations.json
    POST /save                      → write JSON body to annotations.json
    GET  /list                      → list scenes + images under input/
    POST /create-scene?name=…       → mkdir input/<name>/
    POST /upload?scene=…&filename=… → save raw body to input/<scene>/<filename>
    DELETE /image?scene=…&file=…    → delete input/<scene>/<file>
Anything else is served as a static file from this directory.
"""
import http.server
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE  = Path(__file__).resolve().parent
ANN   = HERE / "annotations.json"
INPUT = HERE / "input"

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

def _safe_seg(s: str) -> bool:
    return bool(s) and SAFE_NAME.match(s) and ".." not in s


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    # ── helpers ────────────────────────────────────────────────────────
    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _empty(self, code):
        self.send_response(code); self.end_headers()

    # ── GET ────────────────────────────────────────────────────────────
    def do_GET(self):
        url = urlparse(self.path)
        p = url.path.rstrip("/") or "/"

        if p in ("/annotations.json",):
            data = ANN.read_bytes() if ANN.exists() else b'{"annotations": []}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data); return

        if p == "/list":
            scenes = []
            if INPUT.exists():
                for d in sorted(INPUT.iterdir()):
                    if not d.is_dir() or d.name.startswith("."): continue
                    imgs = []
                    for f in sorted(d.iterdir()):
                        if f.suffix.lower() in IMG_EXT and f.is_file():
                            st = f.stat()
                            imgs.append({
                                "filename": f.name,
                                "size": st.st_size,
                                "mtime": int(st.st_mtime),
                                "url": f"input/{d.name}/{f.name}",
                            })
                    scenes.append({"name": d.name, "images": imgs})
            self._json(200, {"scenes": scenes}); return

        super().do_GET()

    # ── POST ───────────────────────────────────────────────────────────
    def do_POST(self):
        url = urlparse(self.path)
        p = url.path.rstrip("/") or "/"
        q = parse_qs(url.query)

        if p == "/save":
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n)
            try:
                parsed = json.loads(body or b"{}")
            except json.JSONDecodeError as e:
                self.send_error(400, f"bad json: {e}"); return
            ANN.write_text(json.dumps(parsed, indent=2))
            self._empty(204); return

        if p == "/create-scene":
            name = q.get("name", [""])[0]
            if not _safe_seg(name):
                self.send_error(400, "invalid scene name"); return
            (INPUT / name).mkdir(parents=True, exist_ok=True)
            self._json(201, {"name": name}); return

        if p == "/upload":
            scene = q.get("scene", [""])[0]
            filename = q.get("filename", [""])[0]
            if not _safe_seg(scene) or not _safe_seg(filename):
                self.send_error(400, "invalid scene or filename"); return
            ext = Path(filename).suffix.lower()
            if ext not in IMG_EXT:
                self.send_error(400, f"unsupported extension: {ext}"); return
            target_dir = INPUT / scene
            target_dir.mkdir(parents=True, exist_ok=True)
            n = int(self.headers.get("Content-Length", "0"))
            # stream the body to disk in chunks
            CHUNK = 1 << 20
            with open(target_dir / filename, "wb") as out:
                remaining = n
                while remaining > 0:
                    buf = self.rfile.read(min(CHUNK, remaining))
                    if not buf: break
                    out.write(buf)
                    remaining -= len(buf)
            self._json(201, {
                "scene": scene, "filename": filename,
                "url": f"input/{scene}/{filename}",
            }); return

        self.send_error(404)

    # ── DELETE ─────────────────────────────────────────────────────────
    def do_DELETE(self):
        url = urlparse(self.path)
        p = url.path.rstrip("/") or "/"
        q = parse_qs(url.query)

        if p == "/image":
            scene = q.get("scene", [""])[0]
            filename = q.get("file", [""])[0]
            if not _safe_seg(scene) or not _safe_seg(filename):
                self.send_error(400, "invalid scene or filename"); return
            target = INPUT / scene / filename
            if not target.exists() or not target.is_file():
                self.send_error(404, "no such file"); return
            target.unlink()
            self._empty(204); return

        self.send_error(404)

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] %s - %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    httpd = http.server.HTTPServer(("0.0.0.0", port), Handler)
    print(f"[serve] MultiPano/ on http://localhost:{port}/")
    print(f"[serve]   /          → index.html (doc)")
    print(f"[serve]   /inputs.html → input gallery")
    print(f"[serve]   annotations → {ANN}")
    print(f"[serve]   input/      → {INPUT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")
