"""The API's robustness layer (sci_jepa.api.guard): GPU queue and wait limits, CUDA OOM recovery and
faults, rate limits, Idempotency-Key, the body-size limit, metrics. CPU only, synthetic corpus."""
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from sci_jepa.api import guard as G  # noqa: E402

FORWARD = {"problem": "Predicting reaction yields from written lab protocols.",
           "method": "Fine-tuning a small language model on the protocol text.",
           "conditions": "About 50,000 Suzuki coupling protocols from US patents, 80/20 split."}
ASK = {"text": "We want to predict reaction yields from lab protocols. We fine-tune a small language model "
               "on the protocol text."}


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from sci_jepa.api.engine import Engine
    from sci_jepa.config import PaperAConfig
    from sci_jepa.models.model import PaperAModel
    from sci_jepa.training.train import save_model
    d = tmp_path_factory.mktemp("api_guard")
    cfg = PaperAConfig().override(synthetic=True, embedder="hash", device="cpu", index_dir=str(d / "idx"))
    save_model(PaperAModel(cfg), cfg, str(d / "model"))
    return Engine(str(d / "model"), device="cpu", index="auto", warm_up=False, log=lambda *a: None).load()


def make_client(engine, **kw):
    from sci_jepa.api.app import create_app
    return TestClient(create_app(engine, load_in_background=False, limits=G.Limits(**kw)))


# ------------------------------------------------------------------ GPU gate (no app)
def test_gate_queue_full_and_wait_timeout():
    lock = threading.RLock()
    gate = G.GpuGate(lock, G.Limits(gpu_queue=0))
    with pytest.raises(G.Unavailable) as e:
        gate.run(lambda: 1)
    assert e.value.code == "overloaded" and e.value.retry_after >= 1
    gate = G.GpuGate(lock, G.Limits(gpu_queue=4, gpu_wait_s=0.1))
    held = threading.Event()
    release = threading.Event()

    def hold():
        with lock:
            held.set()
            release.wait(5)
    t = threading.Thread(target=hold)
    t.start()
    held.wait(5)
    try:
        with pytest.raises(G.Unavailable) as e:
            gate.run(lambda: 1)
        assert e.value.code == "overloaded" and gate.waiting == 0
    finally:
        release.set()
        t.join()
    assert gate.run(lambda: 42) == 42


def test_gate_oom_retry_and_fault():
    gate = G.GpuGate(threading.RLock(), G.Limits())
    calls = []

    def oom_once():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "ok"
    assert gate.run(oom_once) == "ok" and gate.ooms == 1 and gate.oom_recovered == 1

    def oom_always():
        raise RuntimeError("CUDA out of memory.")
    with pytest.raises(G.Unavailable) as e:
        gate.run(oom_always)
    assert e.value.code == "gpu_out_of_memory"
    assert not gate.lock._is_owned()                     # the lock is released after every failure

    seen = []
    gate = G.GpuGate(threading.RLock(), G.Limits(), on_fault=seen.append)
    with pytest.raises(G.Unavailable) as e:
        gate.run(lambda: (_ for _ in ()).throw(RuntimeError("CUDA error: an illegal memory access was encountered")))
    assert e.value.code == "gpu_fault" and seen and gate.faulted
    with pytest.raises(G.Unavailable):                   # every later request is refused at once
        gate.run(lambda: 1)
    with pytest.raises(ValueError):                      # ordinary errors pass through unchanged
        G.GpuGate(threading.RLock(), G.Limits()).run(lambda: int("x"))


def test_buckets():
    assert G.bucket_of("GET", "/v1/health") is None and G.bucket_of("GET", "/metrics") is None
    assert G.bucket_of("POST", "/v1/ask") == "gpu" and G.bucket_of("POST", "/v1/ask/paper") == "gpu"
    assert G.bucket_of("POST", "/v1/sessions/ses_x/answers") == "gpu"
    assert G.bucket_of("POST", "/v1/parse") == "read" and G.bucket_of("GET", "/v1/papers/2403.01234") == "read"
    assert G.route_label("/v1/sessions/ses_abc/answers") == "/v1/sessions/{id}/answers"
    assert G.route_label("/v1/papers/hep-th/9901001") == "/v1/papers/{id}"
    assert G.route_label("/v1/papers/search") == "/v1/papers/search"


# ------------------------------------------------------------------ through the app
def test_health_answers_while_gpu_busy(engine):
    with make_client(engine, gpu_wait_s=0.2) as c:
        held, release = threading.Event(), threading.Event()

        def hold():
            with engine.lock:
                held.set()
                release.wait(10)
        t = threading.Thread(target=hold)
        t.start()
        held.wait(5)
        try:
            assert c.get("/v1/health").status_code == 200
            r = c.get("/v1/ready")
            assert r.status_code == 200 and r.json()["queue"]["max"] == 16
            r = c.post("/v1/predict/forward", json={"parts": FORWARD})
            assert r.status_code == 503 and r.json()["code"] == "overloaded"
            assert int(r.headers["Retry-After"]) >= 1 and r.headers["X-Request-Id"]
        finally:
            release.set()
            t.join()
        assert c.post("/v1/predict/forward", json={"parts": FORWARD}).status_code == 200


def test_oom_in_request_is_503(engine, monkeypatch):
    with make_client(engine) as c:
        def boom(*a, **k):
            raise RuntimeError("CUDA out of memory.")
        monkeypatch.setattr(engine, "rank", boom)
        r = c.post("/v1/predict/forward", json={"parts": FORWARD})
        assert r.status_code == 503 and r.json()["code"] == "gpu_out_of_memory" and r.headers["Retry-After"] == "5"
        assert c.get("/v1/ready").status_code == 200    # an OOM does not make the server unready


def test_rate_limit(engine):
    with make_client(engine, rate_gpu=2, rate_read=3) as c:
        codes = [c.post("/v1/predict/forward", json={"parts": FORWARD}) for _ in range(3)]
        assert [r.status_code for r in codes] == [200, 200, 429]
        assert codes[0].headers["RateLimit-Limit"] == "2" and codes[1].headers["RateLimit-Remaining"] == "0"
        assert codes[2].json()["code"] == "rate_limited" and int(codes[2].headers["Retry-After"]) >= 1
        assert [c.get("/v1/tasks").status_code for _ in range(4)] == [200, 200, 200, 429]   # separate budget
        assert all(c.get("/v1/health").status_code == 200 for _ in range(10))               # never limited
        other = c.post("/v1/predict/forward", json={"parts": FORWARD}, headers={"Authorization": "Bearer k1"})
        assert other.status_code == 200                  # another client (API key) has its own budget


def test_idempotency(engine):
    with make_client(engine) as c:
        h = {"Idempotency-Key": "4f0c7a1e-0000-4000-8000-000000000001"}
        a = c.post("/v1/ask", json=ASK, headers=h)
        b = c.post("/v1/ask", json=ASK, headers=h)
        assert a.status_code == b.status_code == 200 and b.headers.get("Idempotent-Replayed") == "true"
        assert a.json()["session"] == b.json()["session"] and "Idempotent-Replayed" not in a.headers
        assert b.headers["X-Request-Id"] != a.headers["X-Request-Id"]
        r = c.post("/v1/ask", json={**ASK, "goal": "forward"}, headers=h)
        assert r.status_code == 422 and r.json()["code"] == "idempotency_key_reused"
        assert c.post("/v1/ask", json=ASK, headers={"Idempotency-Key": "x"}).status_code == 400
        idem = c.app.state.idempotency
        n = len(idem)
        assert c.delete(f"/v1/sessions/{a.json()['session']}").status_code == 204
        assert len(idem) == n - 1                       # the stored replay held the session's text


def test_body_limit_and_request_id(engine):
    with make_client(engine, max_body_bytes=2048) as c:
        r = c.post("/v1/parse", json={"text": "x " * 3000})
        assert r.status_code == 413 and r.json()["code"] == "too_large" and r.headers["X-Request-Id"]

        def gen():                                       # chunked: no Content-Length
            for _ in range(10):
                yield b"a" * 500
        r = c.post("/v1/parse", content=gen(), headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert c.post("/v1/parse", json={"text": ASK["text"]}).status_code == 200
        assert c.get("/v1/health", headers={"X-Request-Id": "my-id.1"}).headers["X-Request-Id"] == "my-id.1"
        bad = c.get("/v1/health", headers={"X-Request-Id": "a b<script>"}).headers["X-Request-Id"]
        assert bad.startswith("req_")


def test_session_answer_too_long_is_rejected(engine):
    with make_client(engine) as c:
        s = c.post("/v1/ask", json=ASK).json()["session"]
        r = c.post(f"/v1/sessions/{s}/answers", json={"answers": {"conditions": "word " * 400}})
        assert r.status_code == 413


def test_session_cap(engine):
    with make_client(engine, max_sessions=1) as c:
        assert c.post("/v1/ask", json=ASK).status_code == 200
        r = c.post("/v1/ask", json=ASK)
        assert r.status_code == 503 and r.json()["code"] == "too_many_sessions"


def test_metrics(engine):
    with make_client(engine) as c:
        c.post("/v1/predict/forward", json={"parts": FORWARD})
        c.get("/v1/papers/9999.99999")
        m = c.get("/metrics").text
        assert 'sci_jepa_requests_total{route="/v1/predict/forward",method="POST",status="200"} 1' in m
        assert 'route="/v1/papers/{id}"' in m and "sci_jepa_gpu_queue_waiting 0" in m
        assert "sci_jepa_request_seconds_bucket" in m


def test_cors(engine):
    with make_client(engine, cors_origins=("https://dash.example",)) as c:
        r = c.options("/v1/ask", headers={"Origin": "https://dash.example", "Access-Control-Request-Method": "POST"})
        assert r.headers.get("access-control-allow-origin") == "https://dash.example"
        r = c.options("/v1/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        assert r.headers.get("access-control-allow-origin") is None
