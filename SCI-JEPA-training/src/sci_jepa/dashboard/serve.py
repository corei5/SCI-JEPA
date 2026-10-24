"""
Serve the dashboard + graph pages over http://localhost (browsers block
fetch() from file:// pages, so the live dashboard needs a tiny local server).

    from sci_jepa.dashboard.serve import setup_dashboard, start_server
    root = setup_dashboard(".../paper_a_results/dashboard")
    url = start_server(root, port=8765)          # background thread, returns the URL

Or from a terminal:  python -m sci_jepa.dashboard.serve <root> [port]
"""
import functools
import http.server
import json
import os
import shutil
import socketserver
import sys
import threading
import urllib.request

VENDOR = {
    "vis-network.min.js": "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js",
    "chart.umd.min.js": "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js",
}
_SERVERS = {}


def ensure_vendor(root):
    """Download the two JS libraries once so pages also work offline later."""
    vd = os.path.join(root, "vendor")
    os.makedirs(vd, exist_ok=True)
    for name, url in VENDOR.items():
        dst = os.path.join(vd, name)
        if not os.path.exists(dst) or os.path.getsize(dst) < 10_000:
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception as e:                       # noqa: BLE001  (pages fall back to the CDN)
                print(f"[serve] could not download {name}: {e} (pages will use the CDN)")


def setup_dashboard(root):
    os.makedirs(os.path.join(root, "runs"), exist_ok=True)
    os.makedirs(os.path.join(root, "graphs"), exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(__file__), "dashboard.html"), os.path.join(root, "index.html"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "infra.html"), os.path.join(root, "infra.html"))
    ensure_vendor(root)
    ensure_vendor(os.path.join(root, "graphs"))
    for f in ("runs.json", "graphs.json"):
        p = os.path.join(root, f)
        if not os.path.exists(p):
            json.dump([], open(p, "w"))
    return root


def register_graph(root, title, rel_path):
    p = os.path.join(root, "graphs.json")
    items = json.load(open(p)) if os.path.exists(p) else []
    items = [g for g in items if g["path"] != rel_path] + [{"title": title, "path": rel_path}]
    json.dump(items, open(p, "w"), indent=1)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def start_server(root, port=8765):
    """Serve `root` on localhost in a daemon thread (idempotent). Returns the URL."""
    key = (os.path.abspath(root), port)
    if key in _SERVERS:
        return f"http://localhost:{port}/"
    handler = functools.partial(_Quiet, directory=os.path.abspath(root))
    for p in range(port, port + 20):
        try:
            srv = socketserver.ThreadingTCPServer(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("no free port")
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _SERVERS[key] = srv
    return f"http://localhost:{p}/"


if __name__ == "__main__":
    root = setup_dashboard(sys.argv[1] if len(sys.argv) > 1 else "paper_a_results/dashboard")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    handler = functools.partial(_Quiet, directory=os.path.abspath(root))
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as s:
        print(f"dashboard: http://localhost:{port}/   (Ctrl-C to stop)")
        s.serve_forever()
