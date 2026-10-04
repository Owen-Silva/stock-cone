"""Stock Cone server: stdlib only. Serves web/ at / and the API at /api/.

    python server.py               -> http://127.0.0.1:8642
    python server.py --port 8650   -> another port (e.g. a test server next to the real one)
    python server.py --no-forward  -> do not run the live-test logger thread (forward.py)
"""

import json
import mimetypes
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import accuracy
import engine
import forward

HOST, PORT = "127.0.0.1", 8642
WEB_DIR = os.path.join(engine.ROOT, "web")
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

ENGINE = engine.Engine()


class Handler(BaseHTTPRequestHandler):
    server_version = "StockCone/0.1"

    def log_message(self, fmt, *args):  # quieter than the default
        pass

    def _send(self, status, body, ctype, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status, obj):
        body = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._send(status, body, MIME[".json"])

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        t0 = time.time()
        url = urlparse(self.path)
        try:
            if url.path == "/api/forecast":
                self._forecast(parse_qs(url.query))
            elif url.path == "/api/accuracy":
                self._accuracy()
            elif url.path == "/api/forward":
                self._forward()
            elif url.path == "/api/health":
                a, b = ENGINE.progress
                self._json(200, {"ready": ENGINE.ready, "status": ENGINE.status, "progress": [a, b]})
            elif url.path.startswith("/api/"):
                self._json(404, {"error": f"No such API endpoint: {url.path}"})
            else:
                self._static(url.path)
        except BrokenPipeError:
            pass
        if url.path.startswith("/api/"):
            print(f"{self.command} {self.path} {time.time() - t0:.2f}s", flush=True)

    def _forecast(self, qs):
        def one(k):
            v = qs.get(k)
            return v[0] if v else None

        ticker = one("ticker")
        if not ticker:
            return self._json(400, {"error": "Missing required parameter: ticker"})
        try:
            out = ENGINE.forecast(ticker, one("horizon") or 5, one("asof"))
            self._json(200, out)
        except engine.ForecastError as e:
            self._json(e.status, {"error": e.msg})
        except Exception as e:  # never a bare 500 page; say what broke
            traceback.print_exc()
            self._json(500, {"error": f"Internal error: {type(e).__name__}: {e}"})

    def _accuracy(self):
        """The scoreboard. Serves data/accuracy.json; if it is missing, partial or
        over a day old, starts ONE background rebuild and returns status
        "building" with progress and the partial results. Never blocks."""
        try:
            self._json(200, accuracy.payload(ENGINE))
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": f"Accuracy scoreboard failed: {type(e).__name__}: {e}"})

    def _forward(self):
        """The live test (forward.py): status + per-horizon scores. Cheap: reads two small
        JSON files and a directory listing; the logging runs in its own thread."""
        try:
            self._json(200, forward.status())
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": f"Live test status failed: {type(e).__name__}: {e}"})

    def _static(self, path):
        rel = unquote(path).lstrip("/") or "index.html"
        full = os.path.realpath(os.path.join(WEB_DIR, rel))
        root = os.path.realpath(WEB_DIR)
        if os.path.isdir(full):
            full = os.path.join(full, "index.html")
        if not (full == root or full.startswith(root + os.sep)) or not os.path.isfile(full):
            return self._json(404, {"error": f"Not found: {path}"})
        ext = os.path.splitext(full)[1].lower()
        ctype = MIME.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Stock Cone server")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-forward", action="store_true", help="skip the live-test logger thread")
    args = ap.parse_args(argv)
    httpd = ThreadingHTTPServer((HOST, args.port), Handler)
    httpd.daemon_threads = True
    print(f"Stock Cone running at http://{HOST}:{args.port}  (Ctrl+C to stop)", flush=True)

    def warm():
        try:
            ENGINE.warm_up()
        except Exception as e:
            ENGINE.status = f"warm-up failed: {e}"
            print(f"WARM-UP FAILED: {e}", flush=True)

    threading.Thread(target=warm, daemon=True).start()
    if not args.no_forward:
        # live test: waits for the warm-up, logs today's cones, scores old ones; every 6 h
        forward.start_loop(ENGINE, every_s=6 * 3600, say=engine.log)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    sys.exit(main())
