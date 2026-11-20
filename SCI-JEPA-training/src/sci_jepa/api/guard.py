"""
Robustness layer of the API for a server with one GPU (specification sections 1, 7c and 10).

One model on one GPU answers one request at a time. Without this layer, a burst of requests fills
the server's thread pool. Every thread then waits on the GPU lock, /v1/health stops answering,
and a CUDA out-of-memory error surfaces as a 500. This module provides:

  GpuGate          admission control for GPU work: a bounded queue, a maximum wait, and a 503
                   `overloaded` answer with Retry-After when either is exceeded. A CUDA OOM empties the
                   cache and retries once, then answers 503 `gpu_out_of_memory`. A CUDA fault (for
                   example an illegal memory access) poisons the CUDA context: the server marks itself
                   not ready and, with --exit-on-cuda-fault, exits so its supervisor restarts it.
  RateLimiter      per-client fixed windows, IETF RateLimit-* headers, 429 with Retry-After
                   (section 1). GPU endpoints and cheap read endpoints have separate budgets.
  IdempotencyStore Idempotency-Key on POST (section 1): a retry with the same key and body replays the
                   first answer; the same key with another body is 422; a retry while the first
                   request still runs is 409. Entries expire with the 24 h privacy limit, and deleting
                   a session also deletes the stored answers that contain its text.
  BodyLimit        ASGI middleware that rejects request bodies above a byte limit before they are
                   parsed (413). Chunked bodies are counted too.
  Metrics          Prometheus counters and histograms for GET /metrics (section 10, service layer).

All state is in memory and belongs to this one process, which suits one GPU and one worker. A
multi-replica deployment needs shared storage (for example Redis) for the limiter, the idempotency
store and the sessions.
"""
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

REQUEST_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:\-]{8,128}$")

# Routes that run the model (GPU budget). Everything else under /v1 uses the read budget.
GPU_ROUTES = ("/v1/ask", "/v1/predict/", "/v1/consistency", "/v1/edit-check", "/v1/intervention")
UNLIMITED = ("/v1/health", "/v1/ready", "/metrics", "/v1/docs", "/v1/openapi.json")


@dataclass
class Limits:
    max_body_bytes: int = 64 * 1024        # largest valid body is ~20 KB (5,000 characters of text, escaped)
    gpu_queue: int = 16                    # requests waiting for the GPU before 503 overloaded
    gpu_wait_s: float = 30.0               # longest wait for the GPU before 503 overloaded
    rate_gpu: int = 60                     # requests per window and client on model endpoints (0 = off)
    rate_read: int = 600                   # requests per window and client on the other endpoints (0 = off)
    rate_window_s: int = 60
    trust_proxy: bool = False              # client address from X-Forwarded-For (only behind your own proxy)
    max_sessions: int = 10_000
    idempotency_ttl_s: int = 24 * 3600
    idempotency_max: int = 10_000
    exit_on_cuda_fault: bool = False       # exit(3) after a CUDA fault so a supervisor restarts the server
    cors_origins: tuple = field(default_factory=tuple)


class Unavailable(Exception):
    """A temporary refusal: 503 (or the given status) with Retry-After."""

    def __init__(self, code, detail, retry_after=5, status=503, title="Service temporarily unavailable"):
        super().__init__(detail)
        self.code, self.detail, self.retry_after, self.status, self.title = code, detail, int(retry_after), status, title


# ==================================================================== GPU admission
def is_oom(e):
    try:
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:                                    # torch without CUDA support
        pass
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


_FAULT = ("cuda error", "cublas_status", "cudnn_status", "device-side assert", "illegal memory access",
          "misaligned address", "unspecified launch failure", "nccl error")


def is_cuda_fault(e):
    return isinstance(e, RuntimeError) and not is_oom(e) and any(s in str(e).lower() for s in _FAULT)


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


class GpuGate:
    """Runs GPU work one request at a time, with a bounded queue and a maximum wait."""

    def __init__(self, lock, limits, on_fault=None, log=None):
        self.lock, self.L, self.on_fault, self.log = lock, limits, on_fault, log
        self._m = threading.Lock()
        self.waiting = 0
        self.busy = False
        self.ewma_s = 1.0                                 # mean time a request holds the GPU (for Retry-After)
        self.ooms = self.oom_recovered = self.faults = self.rejected = 0
        self.faulted = None

    def retry_after(self):
        return int(min(60, max(1, math.ceil((self.waiting + 1) * self.ewma_s))))

    def run(self, fn):
        """fn() while holding the GPU. Raises Unavailable when the queue is full, the wait is too long,
        the GPU is out of memory twice in a row, or the CUDA context is broken."""
        if self.faulted:
            raise Unavailable("gpu_fault", "The GPU failed earlier; the server is restarting.", 30)
        with self._m:
            if self.waiting >= self.L.gpu_queue:
                self.rejected += 1
                raise Unavailable("overloaded", f"{self.waiting} requests are already waiting for the model; "
                                  "try again shortly.", self.retry_after())
            self.waiting += 1
        try:
            got = self.lock.acquire(timeout=self.L.gpu_wait_s)
        finally:
            with self._m:
                self.waiting -= 1
        if not got:
            with self._m:
                self.rejected += 1
            raise Unavailable("overloaded", f"Waited {self.L.gpu_wait_s:.0f} s for the model; try again shortly.",
                              self.retry_after())
        t0 = time.monotonic()
        self.busy = True
        try:
            try:
                return fn()
            except Exception as e:
                if not is_oom(e):
                    raise
                self.ooms += 1
                _empty_cache()                           # often fragmentation: free the cache and try once more
                if self.log:
                    self.log("[api] CUDA out of memory; cache emptied, retrying once")
                out = fn()
                self.oom_recovered += 1
                return out
        except Exception as e:
            if is_oom(e):
                self.ooms += 1
                _empty_cache()
                raise Unavailable("gpu_out_of_memory", "The model ran out of GPU memory on this request; "
                                  "try again, or with fewer filters or a smaller top_k.", 5) from e
            if is_cuda_fault(e):
                self.faults += 1
                self.faulted = f"{type(e).__name__}: {str(e)[:300]}"
                if self.on_fault:
                    self.on_fault(self.faulted)
                raise Unavailable("gpu_fault", "The GPU failed; the server is restarting.", 30) from e
            raise
        finally:
            self.busy = False
            self.ewma_s = 0.8 * self.ewma_s + 0.2 * (time.monotonic() - t0)
            self.lock.release()


# ==================================================================== rate limits
def bucket_of(method, path):
    """'gpu', 'read' or None (not limited)."""
    if not path.startswith("/v1/") or path.startswith(UNLIMITED):
        return None
    if method == "POST" and (path.startswith(GPU_ROUTES) or (path.startswith("/v1/sessions/")
                                                            and path.endswith("/answers"))):
        return "gpu"
    return "read"


def client_id(request, trust_proxy=False):
    """API key (hashed) when one is sent, else the client address."""
    auth = request.headers.get("Authorization", "")
    if auth[:7].lower() == "bearer " and len(auth) > 7:
        return "key:" + hashlib.sha256(auth[7:].strip().encode()).hexdigest()[:16]
    if trust_proxy:
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            return "ip:" + fwd.split(",")[0].strip()[:64]
    return "ip:" + (request.client.host if request.client else "unknown")


class RateLimiter:
    def __init__(self, limits):
        self.L = limits
        self._m = threading.Lock()
        self.win = {}                                    # (client, bucket) -> [window start, count]
        self.limited = 0

    def hit(self, client, bucket):
        limit = self.L.rate_gpu if bucket == "gpu" else self.L.rate_read
        if limit <= 0:
            return None
        W = self.L.rate_window_s
        now = time.time()
        start = now - now % W
        with self._m:
            e = self.win.get((client, bucket))
            if e is None or e[0] != start:
                if len(self.win) > 100_000:              # drop clients from earlier windows
                    self.win = {k: v for k, v in self.win.items() if v[0] == start}
                e = self.win[(client, bucket)] = [start, 0]
            e[1] += 1
            n = e[1]
            if n > limit:
                self.limited += 1
        return {"limit": limit, "remaining": max(0, limit - n), "reset": max(1, int(math.ceil(start + W - now))),
                "ok": n <= limit}


def rate_headers(r):
    return {"RateLimit-Limit": str(r["limit"]), "RateLimit-Remaining": str(r["remaining"]),
            "RateLimit-Reset": str(r["reset"])}


# ==================================================================== idempotency
class IdempotencyStore:
    def __init__(self, ttl_s, max_entries):
        self.ttl, self.max = ttl_s, max_entries
        self._m = threading.Lock()
        self.d = OrderedDict()                           # scope -> entry

    def _sweep(self, now):
        for k in [k for k, e in self.d.items() if e["expires"] < now]:
            del self.d[k]
        while len(self.d) > self.max:
            k, e = next(iter(self.d.items()))
            if e["state"] == "running":                  # never evict a request that still runs
                break
            self.d.popitem(last=False)

    def begin(self, scope, fingerprint):
        """-> ('new' | 'replay' | 'conflict' | 'running', entry)."""
        now = time.time()
        with self._m:
            self._sweep(now)
            e = self.d.get(scope)
            if e is None:
                e = self.d[scope] = {"state": "running", "fp": fingerprint, "expires": now + self.ttl}
                return "new", e
            if e["fp"] != fingerprint:
                return "conflict", e
            return ("running" if e["state"] == "running" else "replay"), e

    def finish(self, scope, status, headers, body):
        session = None
        if body[:1] == b"{":
            try:
                session = json.loads(body).get("session")
            except ValueError:
                pass
        with self._m:
            e = self.d.get(scope)
            if e is not None:
                e.update(state="done", status=status, headers=headers, body=body, session=session)

    def abort(self, scope):
        with self._m:
            self.d.pop(scope, None)

    def forget_session(self, sid):
        """DELETE /v1/sessions/{id} also deletes the stored answers that hold that session's text."""
        with self._m:
            for k in [k for k, e in self.d.items() if e.get("session") == sid]:
                del self.d[k]

    def __len__(self):
        return len(self.d)


# ==================================================================== body size limit (ASGI)
class BodyLimit:
    """Rejects bodies above max_bytes with 413 before anything parses them. Bodies are buffered (they
    are small by construction) and replayed to the app."""

    def __init__(self, app, max_bytes):
        self.app, self.max = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS", "DELETE"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length")
        if cl is not None and (not cl.isdigit() or int(cl) > self.max):
            return await self._reject(scope, send)
        chunks, size, more = [], 0, True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunks.append(msg.get("body", b""))
            size += len(chunks[-1])
            more = msg.get("more_body", False)
            if size > self.max:
                return await self._reject(scope, send)
        body, sent = b"".join(chunks), False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)

    async def _reject(self, scope, send):
        hdrs = dict(scope.get("headers") or [])
        rid = hdrs.get(b"x-request-id", b"").decode("latin-1")
        rid = rid if REQUEST_ID.match(rid) else f"req_{uuid.uuid4().hex[:16]}"
        body = json.dumps({"type": "https://sci-jepa.org/problems/too_large", "title": "Request body too large",
                           "status": 413, "code": "too_large", "instance": scope.get("path"), "request_id": rid,
                           "detail": f"The body may be at most {self.max} bytes."}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/problem+json"),
                                (b"content-length", str(len(body)).encode()), (b"x-request-id", rid.encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})


# ==================================================================== metrics (Prometheus text format)
_ID_SEGMENTS = [(re.compile(r"^/v1/sessions/[^/]+"), "/v1/sessions/{id}"), (re.compile(r"^/v1/papers/(?!search$).+"),
                "/v1/papers/{id}"), (re.compile(r"^/v1/kus/.+"), "/v1/kus/{id}")]
BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


def route_label(path):
    for rx, label in _ID_SEGMENTS:
        if rx.match(path):
            return label + ("/answers" if label.endswith("{id}") and path.endswith("/answers") else "")
    return path if path.startswith(("/v1/", "/metrics")) else "other"


class Metrics:
    def __init__(self):
        self._m = threading.Lock()
        self.count = defaultdict(int)                    # (route, method, status) -> n
        self.hist = defaultdict(lambda: [0] * (len(BUCKETS) + 1))   # (route, method) -> bucket counts (+Inf)
        self.sum = defaultdict(float)

    def observe(self, route, method, status, seconds):
        with self._m:
            self.count[(route, method, str(status))] += 1
            h = self.hist[(route, method)]
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    h[i] += 1
            h[-1] += 1
            self.sum[(route, method)] += seconds

    def render(self, gauges):
        out = ["# TYPE sci_jepa_requests_total counter"]
        with self._m:
            for (r, m, s), n in sorted(self.count.items()):
                out.append(f'sci_jepa_requests_total{{route="{r}",method="{m}",status="{s}"}} {n}')
            out.append("# TYPE sci_jepa_request_seconds histogram")
            for (r, m), h in sorted(self.hist.items()):
                lab = f'route="{r}",method="{m}"'
                for b, n in zip(BUCKETS, h):
                    out.append(f'sci_jepa_request_seconds_bucket{{{lab},le="{b}"}} {n}')
                out.append(f'sci_jepa_request_seconds_bucket{{{lab},le="+Inf"}} {h[-1]}')
                out.append(f"sci_jepa_request_seconds_sum{{{lab}}} {self.sum[(r, m)]:.6f}")
                out.append(f"sci_jepa_request_seconds_count{{{lab}}} {h[-1]}")
        for name, (kind, value) in sorted(gauges.items()):
            out.append(f"# TYPE sci_jepa_{name} {kind}")
            out.append(f"sci_jepa_{name} {value}")
        return "\n".join(out) + "\n"


def cuda_memory():
    """-> dict of GPU facts or None (no CUDA)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"name": torch.cuda.get_device_name(), "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()), "free_bytes": int(free), "total_bytes": int(total)}
    except Exception:
        return None


def schedule_exit(code=3, delay_s=2.0, log=None):
    """Exit soon (after the current response is sent) so a supervisor restarts the server."""
    if log:
        log(f"[api] exiting with code {code} in {delay_s:.0f}s so the supervisor restarts the server")
    t = threading.Timer(delay_s, os._exit, args=(code,))
    t.daemon = True
    t.start()
