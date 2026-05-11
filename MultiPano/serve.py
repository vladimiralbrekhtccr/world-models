#!/usr/bin/env python3
"""Tiny static server for MultiPano/index.html with annotation persistence.

Run from the MultiPano/ directory:
    python serve.py            # binds 0.0.0.0:8765
    python serve.py 9000       # custom port

Then open http://localhost:8765/  (port-forward through SSH if remote).

GET /annotations.json  → reads MultiPano/annotations.json (or returns {} if absent)
POST /save             → writes the JSON body to MultiPano/annotations.json
Anything else is served as a static file from this directory.
"""
import http.server
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANN  = HERE / "annotations.json"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def do_GET(self):
        if self.path.rstrip("/") in ("/annotations.json", "annotations.json"):
            if ANN.exists():
                data = ANN.read_bytes()
            else:
                data = b'{"annotations": []}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") in ("/save", "save"):
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n)
            try:
                parsed = json.loads(body or b"{}")
            except json.JSONDecodeError as e:
                self.send_error(400, f"bad json: {e}")
                return
            ANN.write_text(json.dumps(parsed, indent=2))
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] %s - %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    httpd = http.server.HTTPServer(("0.0.0.0", port), Handler)
    print(f"[serve] MultiPano/ on http://localhost:{port}/  (annotations → {ANN})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")
