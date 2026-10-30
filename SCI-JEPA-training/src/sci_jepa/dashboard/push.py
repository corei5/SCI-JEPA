#!/usr/bin/env python3
"""
Push a dashboard folder to the hosted dashboard on Vercel (deploy/vercel/), so the whole team can
follow training at one URL instead of each person tunnelling to the cluster.

Training pushes by itself: when SCIJEPA_PUSH_URL and SCIJEPA_PUSH_TOKEN are set, every run (LiveRun in
telemetry.py) starts a background thread that uploads its dashboard folder while it trains and once more
when it ends (start_autopush below; SCIJEPA_AUTOPUSH=0 turns it off). Where compute nodes have no internet
(Grete), `run_paper_a.sh submit` instead starts this script on the login node with --until-idle.

By hand, on a LOGIN node (scripts/push_dashboard.py runs this module; needs only python3, no packages):

    export SCIJEPA_PUSH_URL=https://<project>.vercel.app
    export SCIJEPA_PUSH_TOKEN=<PUSH_TOKEN from the Vercel project>
    python3 scripts/push_dashboard.py                              # ./runs/paper_a_results/dashboard, every 60 s
    python3 scripts/push_dashboard.py /path/to/dashboard --every 120
    python3 scripts/push_dashboard.py --once                       # one push, then exit (cron)
    python3 scripts/push_dashboard.py --until-idle                 # exit once your SLURM jobs are gone
    python3 scripts/push_dashboard.py --squeue-args "-A myproject" # whose jobs the Queue page shows

Each round it re-reads only files whose size or mtime changed, gzips them, and uploads the ones whose
content changed. `history` in runs/*/live.json and archive/*/live.json is thinned to at most --max-history
rows (the last row is always kept) to stay inside the free Upstash/Vercel limits; the local files are never
modified. It also pushes `squeue` output as cluster/squeue.json for the Queue page.
"""
import argparse
import base64
import getpass
import gzip
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SKIP_TOP = {"index.html", "infra.html"}              # served by the deployment itself
SKIP_DIRS = {"vendor"}
BATCH_B64 = 3_500_000                                # Vercel function request bodies max out at 4.5 MB
MAX_RAW = 64 * 1024 * 1024
KEEP_REMOTE = ("__", "cluster/", "archive/", "runs_archive.json")   # --prune never deletes these (import_logs.py)
SQUEUE_FIELDS = ("id", "name", "user", "state", "elapsed", "limit", "nodes", "reason", "partition")
SQUEUE_FORMAT = "%i|%j|%u|%T|%M|%l|%D|%R|%P"


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def wanted(rel):
    parts = rel.split("/")
    if rel in SKIP_TOP or SKIP_DIRS.intersection(parts[:-1]) or rel.endswith(".tmp") or ".tmp" in parts[-1]:
        return False
    return rel.endswith(".json") or (parts[0] == "graphs" and rel.endswith(".html"))


def walk(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            full = os.path.join(d, f)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if wanted(rel):
                yield rel, full


def thin_history(data, max_rows):
    """Evenly spaced subset of data["history"] (always ending with the newest row)."""
    hist = data.get("history") if isinstance(data, dict) else None
    if not isinstance(hist, list) or max_rows <= 0 or len(hist) <= max_rows:
        return data
    stride = -(-len(hist) // max_rows)
    rows = hist[::stride]
    if rows[-1] is not hist[-1]:
        rows.append(hist[-1])
    out = dict(data)
    out["history"] = rows
    out["history_thinned"] = {"rows": len(hist), "stride": stride}
    return out


def payload_bytes(rel, raw, max_history):
    if max_history and rel.startswith(("runs/", "archive/")) and rel.endswith("/live.json"):
        try:
            data = json.loads(raw)
        except ValueError:
            return None                                  # half-written by a non-atomic writer: next round
        return json.dumps(thin_history(data, max_history), separators=(",", ":")).encode()
    return raw


def encode(body):
    """(version, base64 gzip). The version is a content hash, so unchanged content is never re-sent."""
    v = hashlib.sha1(body).hexdigest()[:16]
    return v, base64.b64encode(gzip.compress(body, 6, mtime=0)).decode()


def squeue_snapshot(args_str):
    cmd = ["squeue", "-h", "-o", SQUEUE_FORMAT] + shlex.split(args_str)
    snap = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": socket.gethostname(), "cmd": " ".join(cmd),
            "jobs": [], "error": None}
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        snap["error"] = str(e)
        return snap
    if res.returncode:
        snap["error"] = res.stderr.strip()[:500]
    for line in res.stdout.splitlines():
        cols = line.split("|", len(SQUEUE_FIELDS) - 1)
        if len(cols) == len(SQUEUE_FIELDS):
            snap["jobs"].append(dict(zip(SQUEUE_FIELDS, (c.strip() for c in cols))))
    return snap


def jobs_left(args_str):
    """True while squeue lists a job for args_str (or squeue cannot answer)."""
    snap = squeue_snapshot(args_str)
    return bool(snap["jobs"]) or bool(snap["error"])


class Pusher:
    def __init__(self, url, token, timeout=60):
        self.endpoint = url.rstrip("/") + "/api/push"
        self.headers = {"Authorization": "Bearer " + token, "User-Agent": "scijepa-push/1",
                        "Content-Type": "application/json"}
        self.timeout = timeout

    def _call(self, method, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.endpoint, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            raise RuntimeError("HTTP %d from %s: %s" % (e.code, self.endpoint, detail)) from None

    def manifest(self):
        return self._call("GET")

    def push(self, files, deleted=()):
        """files: [(path, version, b64)], sent in batches under the request size limit."""
        batch, size = [], 0
        for p, v, b64 in files:
            if batch and size + len(b64) > BATCH_B64:
                self._call("POST", {"files": batch, "deleted": []})
                batch, size = [], 0
            batch.append({"p": p, "v": v, "b64": b64})
            size += len(b64)
        if batch or deleted:
            self._call("POST", {"files": batch, "deleted": list(deleted)})


def one_round(root, pusher, remote, stat_cache, opts):
    """Push what changed since the last round. remote: path -> version known to be on the server."""
    out, seen = [], set()
    for rel, full in walk(root):
        seen.add(rel)
        try:
            st = os.stat(full)
        except OSError:
            continue
        key = (st.st_mtime_ns, st.st_size)
        if stat_cache.get(rel) == key or st.st_size > MAX_RAW:
            continue
        try:
            with open(full, "rb") as f:
                body = payload_bytes(rel, f.read(), opts.max_history)
        except OSError:
            continue
        if body is None:
            continue
        v, b64 = encode(body)
        stat_cache[rel] = key
        if remote.get(rel) == v:
            continue
        if len(b64) > BATCH_B64:
            log("skip %s: %.1f MB compressed is over the upload limit (lower --max-history?)" % (rel, len(b64) / 1e6))
            continue
        out.append((rel, v, b64))
    if opts.squeue and shutil.which("squeue"):
        v, b64 = encode(json.dumps(squeue_snapshot(opts.squeue_args), separators=(",", ":")).encode())
        if remote.get("cluster/squeue.json") != v:
            out.append(("cluster/squeue.json", v, b64))
    deleted = []
    if opts.prune:
        deleted = sorted(p for p in remote if not p.startswith(KEEP_REMOTE) and p not in seen)
    if out or deleted:
        pusher.push(out, deleted)
        for p, v, _ in out:
            remote[p] = v
        for p in deleted:
            remote.pop(p, None)
            stat_cache.pop(p, None)
    return out, deleted


class _AutoPush:
    """One background pusher per dashboard folder inside a training process (see start_autopush)."""

    def __init__(self, root, url, token, every):
        self.root, self.every = root, every
        self.opts = argparse.Namespace(max_history=int(os.environ.get("SCIJEPA_PUSH_MAX_HISTORY", 1500)),
                                       squeue=False, squeue_args="", prune=False)
        self.pusher, self.remote, self.stat_cache = Pusher(url, token, timeout=30), None, {}
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.failures, self.lockfile = 0, None
        threading.Thread(target=self._loop, name="scijepa-autopush", daemon=True).start()

    def _leader(self):
        """Shards of one job write into the same folder: only the process holding <root>/.autopush.lock pushes."""
        if self.lockfile is not None:
            return True
        try:
            import fcntl
        except ImportError:                                       # Windows: no shards, push from every process
            return True
        f = open(os.path.join(self.root, ".autopush.lock"), "a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self.lockfile = f
        return True

    def round(self):
        with self.lock:
            if not self._leader():
                return
            if self.remote is None:
                self.remote = {k: v for k, v in self.pusher.manifest().items() if not k.startswith("__")}
            one_round(self.root, self.pusher, self.remote, self.stat_cache, self.opts)

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.round()
                self.failures = 0
            except (RuntimeError, OSError, ValueError) as e:      # never disturb training
                self.failures += 1
                self.stat_cache.clear()
                if self.failures in (1, 5) or self.failures % 50 == 0:
                    log("[autopush] %s: %s (retrying; SCIJEPA_AUTOPUSH=0 turns this off)" % (self.root, e))
            self.stop.wait(min(self.every * (2 ** min(self.failures, 4)), 900))

    def final(self):
        """At exit: push the run's last state (finished / failed) once more."""
        self.stop.set()
        if self.failures < 5:
            try:
                self.round()
            except (RuntimeError, OSError, ValueError):
                pass


_AUTO = {}


def start_autopush(root):
    """Push `root` to the hosted dashboard from a background thread while this process runs, if
    SCIJEPA_PUSH_URL and SCIJEPA_PUSH_TOKEN are set (SCIJEPA_AUTOPUSH=0 disables it). Idempotent."""
    url, token = os.environ.get("SCIJEPA_PUSH_URL"), os.environ.get("SCIJEPA_PUSH_TOKEN")
    if not url or not token or os.environ.get("SCIJEPA_AUTOPUSH", "1") in ("0", "false", "no"):
        return None
    key = os.path.abspath(root)
    if key not in _AUTO:
        _AUTO[key] = _AutoPush(key, url, token, float(os.environ.get("SCIJEPA_PUSH_EVERY", 30)))
        if len(_AUTO) == 1:
            import atexit
            atexit.register(lambda: [a.final() for a in _AUTO.values()])
        log("[autopush] %s -> %s" % (key, url))
    return _AUTO[key]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="runs/paper_a_results/dashboard", help="dashboard folder")
    ap.add_argument("--url", default=os.environ.get("SCIJEPA_PUSH_URL"), help="site URL (env SCIJEPA_PUSH_URL)")
    ap.add_argument("--every", type=float, default=60, help="seconds between rounds (default 60)")
    ap.add_argument("--once", action="store_true", help="push once and exit")
    ap.add_argument("--until-idle", action="store_true",
                    help="exit after a final push once squeue shows none of your jobs (run_paper_a.sh submit)")
    ap.add_argument("--max-history", type=int, default=1500, help="rows of live.json history to keep (0 = all)")
    ap.add_argument("--squeue-args", default="-u " + getpass.getuser(), help='squeue filter (default "-u $USER")')
    ap.add_argument("--no-squeue", dest="squeue", action="store_false", help="do not push the SLURM queue")
    ap.add_argument("--prune", action="store_true", help="also delete hosted files that no longer exist locally")
    opts = ap.parse_args(argv)
    token = os.environ.get("SCIJEPA_PUSH_TOKEN")
    if not opts.url or not token:
        ap.error("set SCIJEPA_PUSH_URL (or --url) and SCIJEPA_PUSH_TOKEN")
    if not os.path.isdir(opts.root):
        ap.error("not a folder: %s" % opts.root)

    pusher = Pusher(opts.url, token)
    remote = {k: v for k, v in pusher.manifest().items() if not k.startswith("__")}
    log("pushing %s -> %s (%d files already hosted, every %gs)" % (os.path.abspath(opts.root), opts.url, len(remote), opts.every))
    stat_cache, failures, idle = {}, 0, 0
    while True:
        try:
            sent, deleted = one_round(opts.root, pusher, remote, stat_cache, opts)
            if sent or deleted:
                kb = sum(len(b) for _, _, b in sent) / 1024
                log("pushed %d file(s), %.0f KB%s" % (len(sent), kb, ", deleted %d" % len(deleted) if deleted else ""))
            failures = 0
        except (RuntimeError, OSError, ValueError) as e:      # network blips must not kill a long-running sync
            failures += 1
            stat_cache.clear()                                # re-check everything once the site is reachable
            log("push failed (%d in a row): %s" % (failures, e))
            if opts.once:
                return 1
        if opts.once:
            return 0
        if opts.until_idle and failures == 0 and not jobs_left(opts.squeue_args):
            idle += 1
            if idle >= 2:                                     # two quiet rounds: the last writes are pushed
                log("no SLURM jobs left; exiting")
                return 0
        else:
            idle = 0
        time.sleep(min(opts.every * (2 ** min(failures, 4)), 900))


if __name__ == "__main__":
    sys.exit(main())
