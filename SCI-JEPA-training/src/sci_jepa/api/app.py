"""
FastAPI app of the Sci-JEPA API, first version (specification draft of 11 Nov 2026, sections 1-7).

In this version: every endpoint of sections 2-6, the input checks of section 7, RFC 9457 errors,
X-Request-Id, in-memory sessions with expiry and DELETE, and (sci_jepa/api/guard.py) the robustness
layer for one GPU: a bounded GPU queue with 503 + Retry-After, CUDA out-of-memory recovery, rate
limits with RateLimit-* headers, Idempotency-Key, a request-body size limit, CORS and GET /metrics.
Not yet: login and API keys (section 8), async jobs, the dashboard (9) and the model-quality
monitoring of section 10.
Run with `sci-jepa api --model <model folder>` (see sci_jepa/api/__main__.py).
"""
import hashlib
import logging
import re
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from sci_jepa.api import guard as G
from sci_jepa.api import schemas as M
from sci_jepa.api import textparts as T
from sci_jepa.api.engine import ATTRIBUTION, NotFound, NotReady

log = logging.getLogger("sci_jepa.api")
RANK_NOTE = "Ranked by fit to published findings; not a fact check."
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$|^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$")
CORPUS_ID = re.compile(r"^p_\d+$")                     # corpus papers without an arXiv ID


# ==================================================================== errors (RFC 9457)
class Problem(Exception):
    def __init__(self, status, title, code=None, detail=None, errors=None, headers=None):
        self.status, self.title, self.code, self.detail, self.errors = status, title, code, detail, errors or []
        self.headers = headers


def _problem_response(request, status, title, code=None, detail=None, errors=None, headers=None):
    body = {"type": f"https://sci-jepa.org/problems/{code}" if code else "about:blank", "title": title,
            "status": status, "instance": request.url.path,
            "request_id": getattr(request.state, "request_id", None)}
    if code:
        body["code"] = code
    if detail:
        body["detail"] = detail
    if errors:
        body["errors"] = errors
    return JSONResponse(body, status_code=status, media_type="application/problem+json", headers=headers)


_PYD_CODE = {"literal_error": "invalid_value", "enum": "invalid_value", "greater_than_equal": "out_of_range",
             "less_than_equal": "out_of_range", "greater_than": "out_of_range", "less_than": "out_of_range",
             "string_too_long": "too_long", "string_too_short": "too_short"}


def _validation_problem(request, exc):
    errors, statuses = [], []
    for e in exc.errors():
        loc = [str(x) for x in e.get("loc", ()) if x not in ("body", "query", "path")]
        t, msg = e.get("type", ""), str(e.get("msg", ""))
        code = _PYD_CODE.get(t)
        if t == "value_error":                           # our validators: "code: message"
            m = re.match(r"Value error, (\w+): (.*)", msg)
            code, msg = (m.group(1), m.group(2)) if m else ("invalid_value", msg)
        status = 413 if code == "too_long" else 422 if code else 400
        statuses.append(status)
        errors.append({"field": ".".join(loc) or None, "code": code or ("unknown_field" if t == "extra_forbidden"
                       else "missing" if t == "missing" else "invalid_type"), "message": msg,
                       "hint": _hint(code, e)})
    # too long wins: in a union (text or a change object) the other branch also reports a type error
    status = 413 if 413 in statuses else 400 if 400 in statuses else 422
    return _problem_response(request, status, "The request is not valid", "invalid_request" if status == 400 else
                             errors[0]["code"], errors=errors)


def _hint(code, e):
    ctx = e.get("ctx") or {}
    if code == "invalid_value" and "expected" in ctx:
        return f"Use one of: {ctx['expected']}."
    if code == "out_of_range":
        return "Check the allowed range in the specification (top_k 1-50, years 1990-2026)."
    if code == "too_long" and "max_length" in ctx:
        return f"Shorten it to at most {ctx['max_length']} characters."
    if code == "too_short" and "min_length" in ctx:
        return f"Write at least {ctx['min_length']} characters: a few sentences on your problem, method and result."
    return None


# ==================================================================== sessions (in memory)
class Sessions:
    """Each session carries its own lock ("_lock"): two follow-up answers to one session never interleave."""

    def __init__(self, ttl, max_sessions=10_000):
        self.ttl, self.max, self.d, self.lock = ttl, max_sessions, {}, threading.Lock()

    def create(self, **kw):
        sid = "ses_" + secrets.token_urlsafe(12)
        now = time.time()
        with self.lock:
            self._sweep(now)
            if len(self.d) >= self.max:
                raise G.Unavailable("too_many_sessions", "The server holds too many open sessions; try again later "
                                    "or delete finished sessions.", 60)
            s = self.d[sid] = {"id": sid, "created": now, "expires": now + self.ttl, "last": None,
                               "_lock": threading.Lock(), **kw}
        return s

    def __len__(self):
        return len(self.d)

    def get(self, sid):
        with self.lock:
            s = self.d.get(sid)
            if s is None or s["expires"] < time.time():
                self.d.pop(sid, None)
                raise NotFound(sid)
            return s

    def delete(self, sid):
        with self.lock:
            if self.d.pop(sid, None) is None:
                raise NotFound(sid)

    def _sweep(self, now):
        for k in [k for k, s in self.d.items() if s["expires"] < now]:
            del self.d[k]


def _iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ==================================================================== app
def create_app(engine, session_ttl=24 * 3600, load_in_background=True, limits=None):
    L = limits or G.Limits()

    def on_fault(msg):                                   # a broken CUDA context: not ready from now on
        engine.state["error"] = f"gpu_fault: {msg}"
        log.error("CUDA fault, the server is no longer ready: %s", msg)
        if L.exit_on_cuda_fault:
            G.schedule_exit(3, log=log.error)

    gate = G.GpuGate(engine.lock, L, on_fault=on_fault, log=log.warning)
    limiter, idem, metrics = G.RateLimiter(L), G.IdempotencyStore(L.idempotency_ttl_s, L.idempotency_max), G.Metrics()

    @asynccontextmanager
    async def lifespan(app):
        try:                                             # requests waiting for the GPU must not take every worker
            import anyio.to_thread                       # thread: keep threads free for health and cheap endpoints
            lim = anyio.to_thread.current_default_thread_limiter()
            lim.total_tokens = max(lim.total_tokens, L.gpu_queue + 24)
        except Exception:
            pass
        if load_in_background:                           # /v1/health answers at once; /v1/ready is 503 until loaded
            threading.Thread(target=lambda: _safe_load(engine), daemon=True, name="sci-jepa-load").start()
        yield

    app = FastAPI(title="Sci-JEPA API", version="1.0.0-draft",
                  description="Describe research in a few sentences and get the most fitting published findings, "
                              "with sources. First version: no authentication.",
                  docs_url="/v1/docs", openapi_url="/v1/openapi.json", redoc_url=None, lifespan=lifespan)
    app.state.engine, app.state.sessions = engine, Sessions(session_ttl, L.max_sessions)
    app.state.gate, app.state.limiter, app.state.idempotency, app.state.metrics = gate, limiter, idem, metrics
    app.state.limits = L

    # ---------------------------------------------------------------- middleware and handlers
    @app.middleware("http")
    async def front_door(request: Request, call_next):
        """Request ID, rate limit, Idempotency-Key, metrics and the access log (never the user's text)."""
        rid = request.headers.get("X-Request-Id", "")
        request.state.request_id = rid if G.REQUEST_ID.match(rid) else f"req_{uuid.uuid4().hex[:16]}"
        t0, path, extra = time.monotonic(), request.url.path, {}
        client = G.client_id(request, L.trust_proxy)
        try:
            resp = None
            bucket = G.bucket_of(request.method, path)
            r = limiter.hit(client, bucket) if bucket else None
            if r:
                extra.update(G.rate_headers(r))
                if not r["ok"]:
                    resp = _problem_response(request, 429, "Too many requests", "rate_limited",
                                             f"At most {r['limit']} requests per {L.rate_window_s} s on these endpoints.",
                                             headers={"Retry-After": str(r["reset"])})
            key = request.headers.get("Idempotency-Key")
            if resp is None and key is not None and request.method == "POST":
                resp = await _idempotent(request, call_next, client, key)
            if resp is None:
                resp = await call_next(request)
        except Exception:
            log.exception("unhandled error in %s", path)
            resp = _problem_response(request, 500, "Internal error", "internal_error",
                                     f"Quote request ID {request.state.request_id} when reporting this.")
        resp.headers["X-Request-Id"] = request.state.request_id
        for k, v in extra.items():
            resp.headers[k] = v
        if engine.state["model_loaded"]:
            resp.headers["X-Model-Id"] = engine.model_id
        dt = time.monotonic() - t0
        metrics.observe(G.route_label(path), request.method, resp.status_code, dt)
        log.info("%s %s %s %.0fms %s", request.method, path, resp.status_code, 1000 * dt, request.state.request_id)
        return resp

    async def _idempotent(request, call_next, client, key):
        if not G.IDEMPOTENCY_KEY.match(key):
            return _problem_response(request, 400, "Invalid Idempotency-Key", "invalid_value", errors=[{
                "field": "Idempotency-Key", "code": "invalid_value", "message": "8 to 128 characters: letters, digits, "
                "'.', '_', ':' or '-'.", "hint": "Send a UUID."}])
        body = await request.body()
        scope = (client, request.url.path, key)
        state, e = idem.begin(scope, hashlib.sha256(body).hexdigest())
        if state == "replay":
            r = Response(e["body"], status_code=e["status"], headers=e["headers"])
            r.headers["Idempotent-Replayed"] = "true"
            return r
        if state == "conflict":
            return _problem_response(request, 422, "Idempotency-Key reused with another body", "idempotency_key_reused",
                                     "Use a new key for a new request.")
        if state == "running":
            return _problem_response(request, 409, "The first request with this Idempotency-Key is still running",
                                     "idempotency_in_flight", headers={"Retry-After": "1"})
        try:
            resp = await call_next(request)
            data = b"".join([c async for c in resp.body_iterator])
        except Exception:
            idem.abort(scope)
            raise
        hdrs = {k: v for k, v in resp.headers.items() if k.lower() not in ("content-length", "x-request-id")}
        if resp.status_code < 500 and resp.status_code != 429:
            idem.finish(scope, resp.status_code, hdrs, data)
        else:                                            # a server-side failure: let the retry run again
            idem.abort(scope)
        return Response(data, status_code=resp.status_code, headers=hdrs)

    app.add_middleware(G.BodyLimit, max_bytes=L.max_body_bytes)   # outermost: before anything reads the body
    if L.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=list(L.cors_origins), allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["Content-Type", "Authorization", "Idempotency-Key", "X-Request-Id"],
                           expose_headers=["X-Request-Id", "X-Model-Id", "Retry-After", "RateLimit-Limit",
                                           "RateLimit-Remaining", "RateLimit-Reset", "Idempotent-Replayed"],
                           allow_credentials=True, max_age=600)

    @app.exception_handler(RequestValidationError)
    async def _v(request, exc):
        return _validation_problem(request, exc)

    @app.exception_handler(Problem)
    async def _p(request, exc):
        return _problem_response(request, exc.status, exc.title, exc.code, exc.detail, exc.errors, exc.headers)

    @app.exception_handler(G.Unavailable)
    async def _u(request, exc):
        return _problem_response(request, exc.status, exc.title, exc.code, exc.detail,
                                 headers={"Retry-After": str(exc.retry_after)})

    @app.exception_handler(NotReady)
    async def _nr(request, exc):
        return _problem_response(request, 503, "Not available on this server", "not_ready", str(exc),
                                 headers={"Retry-After": "10"})

    @app.exception_handler(NotFound)
    async def _nf(request, exc):
        return _problem_response(request, 404, "Not found", "not_found", f"{exc.args[0]!r} does not exist or expired")

    @app.exception_handler(StarletteHTTPException)
    async def _h(request, exc):
        return _problem_response(request, exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _e(request, exc):
        log.exception("unhandled error in %s", request.url.path)
        return _problem_response(request, 500, "Internal error", "internal_error",
                                 f"Quote request ID {getattr(request.state, 'request_id', '?')} when reporting this.")

    # ---------------------------------------------------------------- helpers
    def need_model(pin=None):
        if gate.faulted:
            raise G.Unavailable("gpu_fault", "The GPU failed; the server is restarting.", 30)
        if not engine.state["model_loaded"]:
            raise NotReady("the model is still loading" if not engine.state["error"] else engine.state["error"])
        if pin and pin != engine.model_id:
            raise Problem(422, "Unknown model", "invalid_value", errors=[{
                "field": "model", "code": "invalid_value", "message": f"{pin!r} is not served here",
                "hint": f"Use {engine.model_id!r} or leave `model` out."}])

    def opts(o):
        f = o.filters.model_dump() if o.filters else None
        return o.top_k, o.scoring, f

    def wrap(body, warnings=(), notes=(RANK_NOTE,)):
        return {**body, "warnings": list(warnings), "notes": list(notes), "model": engine.model_ref(),
                "attribution": ATTRIBUTION}

    def fail_checks(checks, prefix="parts"):
        bad = [c for c in checks if not c["ok"]]
        raise Problem(422, "The input is incomplete or too vague", bad[0]["reason"], errors=[
            {"field": f"{prefix}.{c['part']}", "code": c["reason"], "message": c["question"],
             "hint": f"For example: {c['example']}" if c.get("example") else None} for c in bad])

    def change_of(parts):
        ch = parts.get("change")
        if isinstance(ch, M.Change):
            ch = ch.model_dump()
        if isinstance(ch, str):
            ch = {"part": "method", "text": ch}
        return ch

    def check_change(parts, ch):
        if not ch or ch.get("part") not in ("method", "conditions"):
            return {"part": "change", "ok": False, "reason": "invalid_change",
                    "question": "Name what you would change: the method or the conditions.",
                    "example": T.QUESTIONS["change"][1]}
        if T.normalise(ch.get("text", "")).casefold() == T.normalise(parts.get(ch["part"], "")).casefold():
            return {"part": "change", "ok": False, "reason": "invalid_change",
                    "question": f"The new {ch['part']} is the same as the current one; what would you change?",
                    "example": T.QUESTIONS["change"][1]}
        return None

    def validate(task, parts, conf=None):
        flat = {k: (v.model_dump() if isinstance(v, M.Change) else v) for k, v in parts.items()}
        ok, checks, warns = T.validate_parts(task, flat, conf)
        if task == "intervention" and flat.get("change"):
            bad = check_change(flat, change_of(flat))
            if bad:
                checks = [bad if c["part"] == "change" else c for c in checks]
                ok = False
        return ok, checks, warns

    def run(task, parts, top_k=10, scoring="hybrid", filters=None):
        """Complete, checked parts -> the task's result body and warnings. The model runs through the GPU gate."""
        if task not in T.RANKING_TASKS + ("consistency", "edit_check", "intervention"):
            raise Problem(422, "Unknown task", "invalid_value")
        return gate.run(lambda: _run(task, parts, top_k, scoring, filters))

    def _run(task, parts, top_k, scoring, filters):
        if task in T.RANKING_TASKS:
            answers, warn, _ = engine.rank(task, parts, top_k, scoring, filters)
            return {"answers": answers}, warn, (RANK_NOTE,)
        if task == "consistency":
            out, warn = engine.consistency(parts["claim"], parts["evidence"], filters=filters)
            return out, warn, ("Probabilities come from the model's relation head; check the sources.",)
        if task == "edit_check":
            out = engine.edit_check(parts["original"], parts["edited"])
            return out, [], tuple(out.pop("notes"))
        out, warn = engine.intervention(parts, change_of(parts), top_k, scoring, filters)
        return out, warn, (RANK_NOTE,)

    # ---------------------------------------------------------------- 2. system
    # health, ready and metrics are `async def`: they run on the event loop, not in the thread pool, so they
    # answer even while every worker thread waits for the GPU.
    @app.get("/v1/health", tags=["system"])
    async def health():
        return {"status": "ok"}

    @app.get("/v1/ready", tags=["system"])
    async def ready():
        s = engine.state
        index_wanted = engine.index_spec not in (None, "", "none")
        ok = (s["model_loaded"] and not s["error"] and not gate.faulted
              and (not index_wanted or (s["index_loaded"] and s["search_loaded"])))
        body = {"status": "ready" if ok else ("failed" if s["error"] else "loading"),
                "model_loaded": s["model_loaded"], "index_loaded": s["index_loaded"],
                "search_loaded": s["search_loaded"],
                "queue": {"waiting": gate.waiting, "max": L.gpu_queue, "busy": gate.busy}}
        gpu = G.cuda_memory()
        if gpu:
            body["gpu"] = gpu
        if s["error"]:
            body["error"] = s["error"]
        return JSONResponse(body, status_code=200 if ok else 503, headers=None if ok else {"Retry-After": "10"})

    @app.get("/metrics", include_in_schema=False)
    async def prometheus():
        s = engine.state
        g = {"ready": ("gauge", int(bool(s["model_loaded"] and s["search_loaded"] and not s["error"]))),
             "gpu_queue_waiting": ("gauge", gate.waiting), "gpu_queue_max": ("gauge", L.gpu_queue),
             "gpu_busy": ("gauge", int(gate.busy)), "gpu_seconds_per_request_ewma": ("gauge", round(gate.ewma_s, 4)),
             "gpu_oom_total": ("counter", gate.ooms), "gpu_oom_recovered_total": ("counter", gate.oom_recovered),
             "gpu_faults_total": ("counter", gate.faults), "gpu_rejected_total": ("counter", gate.rejected),
             "rate_limited_total": ("counter", limiter.limited), "sessions_active": ("gauge", len(app.state.sessions)),
             "idempotency_entries": ("gauge", len(idem))}
        mem = G.cuda_memory()
        if mem:
            for k in ("allocated_bytes", "reserved_bytes", "free_bytes", "total_bytes"):
                g[f"cuda_memory_{k}"] = ("gauge", mem[k])
        return PlainTextResponse(metrics.render(g), media_type="text/plain; version=0.0.4")

    @app.get("/v1/info", tags=["system"])
    def info():
        need_model()
        return engine.info()

    @app.get("/v1/models", tags=["system"])
    def models():
        need_model()
        return {"models": [{"id": engine.model_id, "default": True, "weights": engine.weights,
                            "sha256": engine.weights_sha256[:16]}]}

    @app.get("/v1/tasks", tags=["system"])
    def tasks():
        return {"tasks": {k: {"question": v["question"], "requires": v["requires"]} for k, v in T.TASKS.items()},
                "parts": T.PARTS}

    # ---------------------------------------------------------------- 3. main flow
    def understood(parts, conf):
        out = []
        for p, v in parts.items():
            if isinstance(v, M.Change):
                v = v.model_dump()
            e = {"part": p, "text": v}
            if p in conf:
                e["confidence"] = conf[p]
            out.append(e)
        return out

    def advance(s, warnings=()):
        task = s["goal"] or T.pick_task({k: v for k, v in s["parts"].items() if k in T.PARTS})
        s["task"] = task
        ok, checks, warns = validate(task, s["parts"], s["conf"])
        warns = list(warnings) + warns
        head = {"session": s["id"], "expires_at": _iso(s["expires"]), "task": task,
                "understood": understood(s["parts"], s["conf"])}
        if not ok:
            body = {"status": "need_more", **head, "missing": [
                {"part": c["part"], "reason": c["reason"], "question": c["question"], "example": c["example"]}
                for c in checks if not c["ok"]], "warnings": warns}
            s["last"] = body
            return body
        o = s["options"]
        parts = {k: (v.model_dump() if isinstance(v, M.Change) else v) for k, v in s["parts"].items()}
        result, w, notes = run(task, parts, o["top_k"], o["scoring"], o["filters"])
        body = {"status": "ok", **head, **wrap(result, warns + w, notes)}
        s["last"] = body
        return body

    @app.post("/v1/ask", tags=["main flow"])
    def ask(req: M.AskRequest):
        need_model(req.model)
        text, pii = T.scrub_pii(req.text)
        warns = [{"code": "pii_removed", "part": None, "message": "Emails or phone numbers were removed."}] if pii else []
        if len(T.sentences(text)) < 2:
            raise Problem(422, "Please write at least two sentences", "too_short", errors=[{
                "field": "text", "code": "too_short", "message": "Write at least 2 sentences.",
                "hint": "Describe your problem, what you did, and (if you have it) what came out."}])
        if not T.is_english(text):
            raise Problem(422, "Unsupported language", "unsupported_language", errors=[{
                "field": "text", "code": "unsupported_language",
                "message": "The model was trained on English papers; please write in English.", "hint": None}])
        if T.is_gibberish(text):
            raise Problem(422, "Unreadable text", "invalid_text", errors=[{
                "field": "text", "code": "invalid_text", "message": "Please enter a readable description.", "hint": None}])
        found = T.split_parts(text, req.goal)
        if not found or all(e["confidence"] < 0.5 for e in found):
            raise Problem(422, "No research content found", "not_scientific", errors=[{
                "field": "text", "code": "not_scientific", "message": "Describe a problem, method or result.",
                "hint": T.QUESTIONS["problem"][1]}])
        top_k, scoring, filters = opts(req.options)
        s = app.state.sessions.create(goal=req.goal, task=None, parts={e["part"]: e["text"] for e in found},
                                      conf={e["part"]: e["confidence"] for e in found},
                                      options={"top_k": top_k, "scoring": scoring, "filters": filters})
        return advance(s, warns)

    @app.post("/v1/sessions/{session_id}/answers", tags=["main flow"])
    def session_answers(session_id: str, req: M.SessionAnswers):
        need_model()
        s = app.state.sessions.get(session_id)
        allowed = set(T.PARTS) | set(T.EXTRA_PARTS)
        bad = [k for k in list(req.answers) + list(req.corrections) if k not in allowed]
        if bad:
            raise Problem(422, "Unknown part", "invalid_value", errors=[
                {"field": f"answers.{k}", "code": "invalid_value", "message": f"{k!r} is not a part",
                 "hint": f"Use one of: {sorted(allowed)}."} for k in bad])
        if not s["_lock"].acquire(timeout=L.gpu_wait_s):
            raise Problem(409, "This session is busy with an earlier answer", "session_busy",
                          headers={"Retry-After": "1"})
        try:
            for src in (req.answers, req.corrections):
                for k, v in src.items():
                    v = T.normalise(v) if isinstance(v, str) else v
                    s["parts"][k] = v
                    s["conf"].pop(k, None)               # typed by the user: no splitting confidence
            if req.options:
                top_k, scoring, filters = opts(req.options)
                s["options"] = {"top_k": top_k, "scoring": scoring, "filters": filters}
            return advance(s)
        finally:
            s["_lock"].release()

    @app.get("/v1/sessions/{session_id}", tags=["main flow"])
    def session_get(session_id: str):
        s = app.state.sessions.get(session_id)
        task = s["task"]
        _, checks, _ = validate(task, s["parts"], s["conf"]) if task else (None, [], None)
        return {"session": s["id"], "expires_at": _iso(s["expires"]), "task": task,
                "understood": understood(s["parts"], s["conf"]),
                "missing": [{"part": c["part"], "reason": c["reason"], "question": c["question"],
                             "example": c["example"]} for c in checks if not c["ok"]],
                "last": s["last"]}

    @app.delete("/v1/sessions/{session_id}", status_code=204, tags=["main flow"])
    def session_delete(session_id: str):
        app.state.sessions.delete(session_id)
        idem.forget_session(session_id)                  # stored idempotent replays hold the same text
        return Response(status_code=204)

    # ---------------------------------------------------------------- 4. building blocks
    @app.post("/v1/parse", tags=["building blocks"])
    def parse(req: M.ParseRequest):
        text, pii = T.scrub_pii(req.text)
        warns = [{"code": "pii_removed", "part": None, "message": "Emails or phone numbers were removed."}] if pii else []
        parts = T.split_parts(text, req.goal)
        warns += [{"code": "low_confidence", "part": e["part"], "message": f"Did we read your {e['part']} correctly?"}
                  for e in parts if e["confidence"] < T.LOW_CONFIDENCE]
        return {"parts": parts, "language": "en" if T.is_english(text) else "unknown", "warnings": warns}

    @app.post("/v1/validate", tags=["building blocks"])
    def validate_ep(req: M.ValidateRequest):
        parts = {k: v for k, v in req.parts.model_dump().items() if v}
        if req.parts.change is not None:
            parts["change"] = req.parts.change
        ok, checks, warns = validate(req.task, parts)
        return {"valid": ok, "task": req.task, "checks": checks, "warnings": warns}

    # ---------------------------------------------------------------- 5. task endpoints
    def structured(task, req):
        need_model(req.model)
        parts = req.parts.given()
        ok, checks, warns = validate(task, parts)
        if not ok:
            fail_checks(checks)
        top_k, scoring, filters = opts(req.options)
        result, w, notes = run(task, parts, top_k, scoring, filters)
        return {"status": "ok", "task": task, **wrap(result, warns + w, notes)}

    @app.post("/v1/predict/forward", tags=["tasks"])
    def predict_forward(req: M.TaskRequest):
        return structured("forward", req)

    @app.post("/v1/predict/inverse", tags=["tasks"])
    def predict_inverse(req: M.TaskRequest):
        return structured("inverse", req)

    @app.post("/v1/predict/abduction", tags=["tasks"])
    def predict_abduction(req: M.TaskRequest):
        return structured("abduction", req)

    @app.post("/v1/consistency", tags=["tasks"])
    def consistency(req: M.ConsistencyRequest):
        need_model(req.model)
        parts = req.parts.given()
        ok, checks, warns = validate("consistency", parts)
        if not ok:
            fail_checks(checks)
        _, _, filters = opts(req.options)
        out, w = gate.run(lambda: engine.consistency(parts["claim"], parts["evidence"], related=req.related,
                                                     filters=filters))
        return {"status": "ok", **wrap(out, warns + w, ("Probabilities come from the model's relation head; "
                                                        "check the sources.",))}

    @app.post("/v1/edit-check", tags=["tasks"])
    def edit_check(req: M.EditCheckRequest):
        need_model(req.model)
        ok, checks, warns = validate("edit_check", {"original": req.original, "edited": req.edited})
        if not ok:
            fail_checks(checks, prefix="body")
        ctx = req.context.given() if req.context else None
        if ctx:
            ok2, checks2, w2 = validate("forward", ctx)
            if not ok2:
                fail_checks(checks2, prefix="context")
            warns += w2
        out = gate.run(lambda: engine.edit_check(req.original, req.edited, ctx))
        notes = out.pop("notes")
        return {"status": "ok", **wrap(out, warns, notes)}

    @app.post("/v1/intervention", tags=["tasks"])
    def intervention(req: M.InterventionRequest):
        need_model(req.model)
        parts = {**req.parts.given(), "change": req.change}
        ok, checks, warns = validate("intervention", parts)
        if not ok:
            fail_checks(checks)
        top_k, scoring, filters = opts(req.options)
        flat = {**req.parts.given(), "change": req.change.model_dump()}
        result, w, notes = run("intervention", flat, top_k, scoring, filters)
        return {"status": "ok", **wrap(result, warns + w, notes)}

    # ---------------------------------------------------------------- 6. papers and the corpus
    def check_id(pid):
        if not (ARXIV_ID.match(pid) or CORPUS_ID.match(pid)):
            raise Problem(422, "Not an arXiv ID", "invalid_id", errors=[{
                "field": "paper_id", "code": "invalid_id", "message": f"{pid!r} is not an arXiv ID",
                "hint": "Use e.g. 2403.01234 or hep-th/9901001 (or a p_<n> ID from an earlier answer)."}])

    @app.get("/v1/papers/search", tags=["papers"])
    def papers_search(q: str = Query(..., min_length=1, max_length=200), limit: int = Query(20, ge=1, le=100),
                      cursor: str = Query(None, pattern=r"^c_\d+$")):
        need_model()
        items, nxt = engine.search_ids(q.strip(), limit, int(cursor[2:]) if cursor else 0)
        return {"items": items, "next_cursor": f"c_{nxt}" if nxt is not None else None,
                "notes": ["The index holds no titles yet: q matches the start of the arXiv ID."]}

    @app.get("/v1/papers/{paper_id:path}", tags=["papers"])
    def paper(paper_id: str):
        need_model()
        check_id(paper_id)
        return engine.paper(paper_id)

    @app.get("/v1/kus/{ku_id}", tags=["papers"])
    def ku(ku_id: str):
        need_model()
        return engine.ku(ku_id)

    @app.post("/v1/ask/paper", tags=["papers"])
    def ask_paper(req: M.AskPaperRequest):
        need_model(req.model)
        check_id(req.paper_id)
        p = engine.paper_index(req.paper_id)
        meta = engine.paper_meta(p)
        if req.reveal_truth and meta["split"] != "val":
            raise Problem(403, "reveal_truth is allowed only for validation papers", "reveal_not_allowed",
                          detail=f"{req.paper_id} is a {meta['split'] or 'unsplit'} paper; test papers stay hidden "
                                 "until the review of the accompanying paper is complete.")
        rec = engine.paper_record(p)
        parts = engine.parts_of_record(rec)
        ok, checks, warns = validate(req.goal, parts)
        missing = [c for c in checks if not c["ok"] and c["reason"] == "missing_part"]
        if missing:
            raise Problem(422, "This paper lacks a part the task needs", "missing_part", errors=[
                {"field": c["part"], "code": "missing_part", "message": f"The paper has no {c['part']} section.",
                 "hint": "Try another goal."} for c in missing])
        top_k, scoring, filters = opts(req.options)
        truth = None
        if req.reveal_truth:
            from sci_jepa.data.ku_schema import COMP_TYPES
            tt = set(T.TASKS[req.goal]["answers"])
            truth = [int(c) for c in engine.gi.paper_comps[p] if COMP_TYPES[int(engine.gi.comp_type[c])] in tt]
        answers, w, extra = gate.run(lambda: engine.rank(req.goal, parts, top_k, scoring, filters, rec=rec,
                                                         include=(p,), truth=truth))
        body = {"status": "ok", "paper": meta, "task": req.goal,
                "understood": [{"part": k, "text": v} for k, v in parts.items()], "answers": answers}
        if truth is not None:
            body["truth"] = extra.get("truth", [])
        return wrap(body, warns + w)

    return app


def _safe_load(engine):
    try:
        engine.load()
    except Exception:
        log.exception("loading the model or index failed; /v1/ready reports it")
