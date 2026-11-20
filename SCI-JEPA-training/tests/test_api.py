"""The HTTP API (sci_jepa.api) end to end on the synthetic corpus with an untrained model (CPU, ~1 min).
Checks shapes, the input checks and the error format, not answer quality."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from sci_jepa.api import textparts as T  # noqa: E402

FORWARD = {"problem": "Predicting reaction yields from written lab protocols.",
           "method": "Fine-tuning a small language model on the protocol text.",
           "conditions": "About 50,000 Suzuki coupling protocols from US patents, 80/20 split."}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from sci_jepa.api.app import create_app
    from sci_jepa.api.engine import Engine
    from sci_jepa.config import PaperAConfig
    from sci_jepa.models.model import PaperAModel
    from sci_jepa.training.train import save_model
    d = tmp_path_factory.mktemp("api")
    cfg = PaperAConfig().override(synthetic=True, embedder="hash", device="cpu", index_dir=str(d / "idx"))
    save_model(PaperAModel(cfg), cfg, str(d / "model"))
    eng = Engine(str(d / "model"), device="cpu", index="auto", warm_up=False, log=lambda *a: None).load()
    with TestClient(create_app(eng, load_in_background=False)) as c:
        c.engine = eng
        yield c


# ------------------------------------------------------------------ text side (no model)
def test_splitter_finds_problem_and_method():
    parts = {e["part"]: e for e in T.split_parts("We want to predict reaction yields from lab protocols. "
                                                 "We fine-tune a small language model on the protocol text.")}
    assert set(parts) == {"problem", "method"}
    assert parts["problem"]["span"][0] == 0


def test_checks():
    assert T.check_part("result", "Accuracy improved")[0]["reason"] == "too_vague"
    assert T.check_part("result", "Top-1 accuracy rose from 71.2% to 76.8% over a ResNet-50 baseline.")[0] is None
    assert T.check_part("method", "we did experiments on the data")[0]["reason"] == "too_vague"
    bad, warn = T.check_part("result", "Galaxy rotation curves stay flat far beyond the visible disk.")
    assert bad is None and warn[0]["code"] == "result_not_specific"     # qualitative: a warning, not an error
    assert T.check_part("problem", "asdfgh qwrtzp xcvbnm kjhgfd lkjhgf poiuzt")[0]["reason"] == "invalid_text"
    ok, checks, _ = T.validate_parts("forward", {**FORWARD, "method": FORWARD["problem"]})
    assert not ok and any(c.get("reason") == "duplicate_part" for c in checks)
    assert T.edit_kind("Adding dropout reduced overfitting by 12%.", "Adding dropout increased overfitting by 12%.") \
        == "direction"
    assert T.edit_kind("Loss fell by 12%.", "Loss fell by 6%.") == "number"
    assert not T.is_english("Wir wollen die Ausbeute von Reaktionen aus Laborprotokollen vorhersagen und so weiter.")


# ------------------------------------------------------------------ system
def test_system(client):
    assert client.get("/v1/health").json() == {"status": "ok"}
    r = client.get("/v1/ready")
    assert r.status_code == 200 and r.json()["index_loaded"]
    info = client.get("/v1/info").json()
    assert info["corpus"]["knowledge_units"] > 0 and "forward" in info["tasks"]
    assert client.get("/v1/models").json()["models"][0]["default"]
    assert client.get("/v1/tasks").json()["tasks"]["inverse"]["requires"] == ["problem", "result"]
    assert r.headers["X-Request-Id"]
    assert client.get("/v1/health", headers={"X-Request-Id": "abc"}).headers["X-Request-Id"] == "abc"


# ------------------------------------------------------------------ main flow
def test_ask_need_more_then_ok(client):
    r = client.post("/v1/ask", json={"text": "We want to predict reaction yields from lab protocols. "
                                             "We fine-tune a small language model on the protocol text.",
                                     "goal": "forward", "options": {"top_k": 5}})
    b = r.json()
    assert r.status_code == 200 and b["status"] == "need_more", b
    assert [m["part"] for m in b["missing"]] == ["conditions"]
    sid = b["session"]
    b = client.post(f"/v1/sessions/{sid}/answers", json={"answers": {"conditions": FORWARD["conditions"]}}).json()
    assert b["status"] == "ok", b
    assert 1 <= len(b["answers"]) <= 5 and b["answers"][0]["source"]["type"] == "result"
    assert set(b["answers"][0]["scores"]) == {"jepa", "text", "bm25"}
    assert b["model"]["id"].startswith("sci-jepa-") and b["attribution"]
    assert client.get(f"/v1/sessions/{sid}").json()["last"]["status"] == "ok"
    assert client.delete(f"/v1/sessions/{sid}").status_code == 204
    assert client.get(f"/v1/sessions/{sid}").status_code == 404


def test_ask_is_deterministic(client):
    body = {"text": "We want to predict reaction yields from lab protocols. We fine-tune a small language model "
                    "on the protocol text. We use about 50,000 Suzuki coupling protocols from US patents, 80/20 split.",
            "goal": "forward"}
    a, b = client.post("/v1/ask", json=body).json(), client.post("/v1/ask", json=body).json()
    assert a["status"] == "ok", a
    assert [x["source"]["ku_id"] for x in a["answers"]] == [x["source"]["ku_id"] for x in b["answers"]]


def test_ask_errors(client):
    r = client.post("/v1/ask", json={"text": "short"})
    assert r.status_code == 422 and r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["errors"][0]["code"] == "too_short"
    assert client.post("/v1/ask", json={"text": "x" * 5001}).status_code == 413
    assert client.post("/v1/ask", json={"text": "a" * 50, "colour": 1}).status_code == 400
    r = client.post("/v1/ask", json={"text": "We want to predict yields from protocols. We fine-tune a model on it.",
                                     "goal": "telepathy"})
    assert r.status_code == 422 and r.json()["errors"][0]["code"] == "invalid_value"
    r = client.post("/v1/ask", json={"text": "We want to predict yields from protocols. We fine-tune a model on it.",
                                     "options": {"top_k": 99}})
    assert r.status_code == 422 and r.json()["errors"][0]["code"] == "out_of_range"
    r = client.post("/v1/ask", json={"text": "We want to predict yields from protocols. We fine-tune a model on it.",
                                     "options": {"filters": {"year_from": 2020, "year_to": 2010}}})
    assert r.status_code == 422 and r.json()["errors"][0]["code"] == "out_of_range"
    r = client.post("/v1/ask", json={"text": "Wir wollen die Ausbeute vorhersagen. Wir trainieren ein Sprachmodell "
                                             "auf den Protokollen der Reaktionen."})
    assert r.status_code == 422 and r.json()["code"] == "unsupported_language"
    r = client.post("/v1/ask", json={"text": "We want to predict yields from protocols. We fine-tune a model on it.",
                                     "model": "sci-jepa-9.9.9"})
    assert r.status_code == 422


def test_pii_removed(client):
    b = client.post("/v1/parse", json={"text": "We want to predict yields; mail jane.doe@example.org. "
                                               "We fine-tune a small language model."}).json()
    assert any(w["code"] == "pii_removed" for w in b["warnings"])
    assert all("@" not in p["text"] for p in b["parts"])


# ------------------------------------------------------------------ building blocks
def test_validate_matches_spec_example(client):
    b = client.post("/v1/validate", json={"task": "inverse", "parts": {
        "problem": "Image classifiers overfit on small medical datasets.", "result": "Accuracy improved"}}).json()
    assert b["valid"] is False
    assert b["checks"][0] == {"part": "problem", "ok": True}
    assert b["checks"][1]["reason"] == "too_vague"


# ------------------------------------------------------------------ task endpoints
@pytest.mark.parametrize("path,parts,types", [
    ("/v1/predict/forward", FORWARD, {"result"}),
    ("/v1/predict/inverse", {"problem": FORWARD["problem"],
                             "result": "Top-1 accuracy rose from 71.2% to 76.8% over a ResNet-50 baseline."},
     {"method", "procedure"}),
    ("/v1/predict/abduction", {"result": "Top-1 accuracy rose from 71.2% to 76.8% over a ResNet-50 baseline.",
                               "conditions": FORWARD["conditions"]}, {"claim", "question"}),
])
def test_predict(client, path, parts, types):
    r = client.post(path, json={"parts": parts, "options": {"top_k": 4, "scoring": "hybrid"}})
    assert r.status_code == 200, r.text
    ans = r.json()["answers"]
    assert ans and {a["source"]["type"] for a in ans} <= types
    assert [a["rank"] for a in ans] == list(range(1, len(ans) + 1))


def test_predict_missing_part_is_422(client):
    r = client.post("/v1/predict/forward", json={"parts": {"problem": FORWARD["problem"]}})
    assert r.status_code == 422
    assert {e["field"] for e in r.json()["errors"]} == {"parts.method", "parts.conditions"}


def test_filters(client):
    fields = client.get("/v1/info").json()["corpus"]["fields"]
    r = client.post("/v1/predict/forward", json={"parts": FORWARD, "options": {"filters": {"fields": ["Physics"]}}})
    assert r.status_code == 200
    if "Physics" in fields:
        assert all(a["source"]["paper"]["field"] == "Physics" for a in r.json()["answers"])


def test_consistency(client):
    r = client.post("/v1/consistency", json={"parts": {
        "claim": "Dropout reduces overfitting in small transformers.",
        "evidence": "On 5 datasets, validation loss fell by 12% with dropout 0.1 versus none."}})
    b = r.json()
    assert r.status_code == 200, b
    assert b["verdict"] in ("supports", "challenges")
    assert abs(sum(b["probabilities"].values()) - 1) < 1e-3
    assert set(b["related_evidence"]) == {"supports", "challenges"}


def test_edit_check(client):
    b = client.post("/v1/edit-check", json={"original": "Adding dropout reduced overfitting by 12%.",
                                            "edited": "Adding dropout increased overfitting by 12%."}).json()
    assert b["status"] == "ok" and b["edit_kind"] == "direction" and isinstance(b["separated"], bool)
    b = client.post("/v1/edit-check", json={"original": "Adding dropout reduced overfitting by 12%.",
                                            "edited": "Adding dropout reduced overfitting by 6%.",
                                            "context": FORWARD}).json()
    assert b["mode"] == "prediction_prefers_original" and b["edit_kind"] == "number"
    r = client.post("/v1/edit-check", json={"original": "Same text here.", "edited": "Same text here."})
    assert r.status_code == 422


def test_intervention(client):
    r = client.post("/v1/intervention", json={"parts": FORWARD, "change": {
        "part": "method", "text": "Training a graph neural network on reaction graphs."}, "options": {"top_k": 3}})
    b = r.json()
    assert r.status_code == 200, b
    assert 0 <= b["shift"] <= 1 and len(b["before"]) == len(b["after"]) == 3
    r = client.post("/v1/intervention", json={"parts": FORWARD, "change": {"part": "problem", "text": "Something else."}})
    assert r.status_code == 422 and r.json()["errors"][0]["code"] == "invalid_change"


# ------------------------------------------------------------------ papers
def test_papers(client):
    pid = str(client.engine.gi.paper_ids[0])
    p = client.get(f"/v1/papers/{pid}").json()
    assert p["paper_id"] == pid and p["knowledge_units"]
    ku = p["knowledge_units"][0]["ku_id"]
    assert client.get(f"/v1/kus/{ku}").json()["paper"]["paper_id"] == pid
    s = client.get("/v1/papers/search", params={"q": pid[:4], "limit": 2}).json()
    assert len(s["items"]) <= 2 and s["items"][0]["paper_id"].startswith(pid[:4])
    assert client.get("/v1/papers/not-an-id").status_code == 422
    assert client.get("/v1/papers/9999.99999").status_code == 404
    assert client.get("/v1/kus/ku_999999999").status_code == 404
    assert client.get("/v1/papers/p_0").status_code == 404             # p_<n> only for papers without an arXiv ID
    assert all(a["paper"]["arxiv_id"] == a["paper"]["paper_id"] for a in [client.get(f"/v1/kus/{ku}").json()])


def test_ask_paper(client):
    gi = client.engine.gi
    p = next(int(x) for x in gi.papers("train"))
    pid = str(gi.paper_ids[p])
    b = client.post("/v1/ask/paper", json={"paper_id": pid, "goal": "forward", "options": {"top_k": 3}}).json()
    assert b["status"] == "ok" and b["paper"]["paper_id"] == pid, b
    r = client.post("/v1/ask/paper", json={"paper_id": pid, "goal": "forward", "reveal_truth": True})
    assert r.status_code == 403                          # train paper: no truth
