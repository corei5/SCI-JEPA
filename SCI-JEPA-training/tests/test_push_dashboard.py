"""Unit tests of sci_jepa/dashboard/push.py (the cluster -> Vercel sync, scripts/push_dashboard.py); no network."""
import base64
import gzip
import importlib.util
import json
import os
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location(
    "push_dashboard", os.path.join(os.path.dirname(__file__), "..", "src", "sci_jepa", "dashboard", "push.py"))
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)

OPTS = SimpleNamespace(max_history=10, squeue=False, squeue_args="", prune=False)


class FakePusher:
    def __init__(self):
        self.calls = []

    def push(self, files, deleted=()):
        self.calls.append((list(files), list(deleted)))


def _dash(tmp_path, n_hist=100):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (tmp_path / "vendor").mkdir()
    (tmp_path / "graphs").mkdir()
    (tmp_path / "index.html").write_text("<html>")
    (tmp_path / "vendor" / "chart.umd.min.js").write_text("x")
    (tmp_path / "graphs" / "codebase.html").write_text("<html>graph")
    (tmp_path / "runs.json").write_text(json.dumps([{"name": "r1", "path": "runs/r1/live.json"}]))
    (run / "live.json").write_text(json.dumps({"name": "r1", "history": [{"step": i} for i in range(n_hist)]}))
    (run / "live.json.tmp").write_text("{half")
    return tmp_path


def test_selects_data_files_only(tmp_path):
    root = _dash(tmp_path)
    assert sorted(rel for rel, _ in pd.walk(str(root))) == ["graphs/codebase.html", "runs.json", "runs/r1/live.json"]


def test_thin_history_keeps_last_row_and_bounds_size():
    data = {"history": [{"step": i} for i in range(1001)], "evals": [1]}
    out = pd.thin_history(data, 100)
    assert len(out["history"]) <= 101 and out["history"][0]["step"] == 0 and out["history"][-1]["step"] == 1000
    assert out["history_thinned"]["rows"] == 1001 and out["evals"] == [1]
    assert len(data["history"]) == 1001                         # the input is not modified
    assert pd.thin_history({"history": [1, 2]}, 100) == {"history": [1, 2]}


def test_round_pushes_changes_once(tmp_path):
    root = _dash(tmp_path)
    pusher, remote, cache = FakePusher(), {}, {}
    sent, _ = pd.one_round(str(root), pusher, remote, cache, OPTS)
    assert {p for p, _, _ in sent} == {"graphs/codebase.html", "runs.json", "runs/r1/live.json"}
    live = next(b for p, _, b in sent if p == "runs/r1/live.json")
    hosted = json.loads(gzip.decompress(base64.b64decode(live)))
    assert len(hosted["history"]) <= 11 and hosted["history"][-1]["step"] == 99

    assert pd.one_round(str(root), pusher, remote, cache, OPTS) == ([], [])     # nothing changed

    os.utime(root / "runs.json", ns=(1, 1))                                    # touched, same content
    assert pd.one_round(str(root), pusher, remote, cache, OPTS) == ([], [])

    (root / "runs" / "r1" / "live.json").write_text(json.dumps({"name": "r1", "history": [{"step": 0}]}))
    sent, _ = pd.one_round(str(root), pusher, remote, cache, OPTS)
    assert [p for p, _, _ in sent] == ["runs/r1/live.json"]


def test_restart_skips_files_already_hosted(tmp_path):
    root = _dash(tmp_path)
    remote = {}
    pd.one_round(str(root), FakePusher(), remote, {}, OPTS)
    pusher = FakePusher()
    assert pd.one_round(str(root), pusher, dict(remote), {}, OPTS) == ([], []) and not pusher.calls


def test_prune_is_opt_in(tmp_path):
    root = _dash(tmp_path)
    remote = {"runs/gone/live.json": "abc", "cluster/squeue.json": "def",
              "archive/log-x/live.json": "0a", "runs_archive.json": "0b"}
    _, deleted = pd.one_round(str(root), FakePusher(), dict(remote), {}, OPTS)
    assert deleted == []
    _, deleted = pd.one_round(str(root), FakePusher(), dict(remote), {}, SimpleNamespace(**{**vars(OPTS), "prune": True}))
    assert deleted == ["runs/gone/live.json"]


def test_squeue_parsing(monkeypatch):
    out = "123|train|me|RUNNING|1:00|2:00:00|1|ggpu01|grete\n124|eval|me|PENDING|0:00|1:00:00|1|(Priority)|grete\n"
    monkeypatch.setattr(pd.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=out, stderr=""))
    snap = pd.squeue_snapshot("-u me")
    assert [j["state"] for j in snap["jobs"]] == ["RUNNING", "PENDING"] and snap["jobs"][1]["reason"] == "(Priority)"


def test_autopush_needs_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("SCIJEPA_PUSH_URL", raising=False)
    assert pd.start_autopush(str(tmp_path)) is None
    monkeypatch.setenv("SCIJEPA_PUSH_URL", "https://x.invalid")
    monkeypatch.setenv("SCIJEPA_PUSH_TOKEN", "t")
    monkeypatch.setenv("SCIJEPA_AUTOPUSH", "0")
    assert pd.start_autopush(str(tmp_path)) is None


def test_autopush_round_and_leader_lock(tmp_path, monkeypatch):
    monkeypatch.delenv("SCIJEPA_PUSH_FOLDER", raising=False)
    root = _dash(tmp_path)
    monkeypatch.setattr(pd.threading, "Thread", lambda **k: SimpleNamespace(start=lambda: None))
    a, b = pd._AutoPush(str(root), "https://x.invalid", "t", 30), pd._AutoPush(str(root), "https://x.invalid", "t", 30)
    for ap in (a, b):
        ap.pusher = FakePusher()
        ap.pusher.manifest = lambda: {"__synced": "now"}
    a.round()
    pre = "d/%s/" % root.name                                   # the dashboard has a runs.json: its own folder
    assert {p for p, _, _ in a.pusher.calls[0][0]} == {pre + "graphs/codebase.html", pre + "runs.json",
                                                         pre + "runs/r1/live.json"}
    b.round()                                                   # a holds .autopush.lock: b stays quiet
    assert not b.pusher.calls


def test_folder_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("SCIJEPA_PUSH_FOLDER", raising=False)
    dash = tmp_path / "arxiv_full_M_e2" / "dashboard"
    dash.mkdir(parents=True)
    assert pd.folder_prefix(str(dash)) == ""                    # no runs.json (an import_logs archive): top level
    (dash / "runs.json").write_text("[]")
    assert pd.folder_prefix(str(dash)) == "d/arxiv_full_M_e2/"   # <results>/dashboard -> d/<results>/
    assert pd.folder_prefix(str(dash), ".") == ""
    assert pd.folder_prefix(str(dash), "my run/x") == "d/my_run_x/"
    monkeypatch.setenv("SCIJEPA_PUSH_FOLDER", "other")
    assert pd.folder_prefix(str(dash)) == "d/other/"


def test_folders_do_not_collide_and_prune_stays_inside(tmp_path):
    root = _dash(tmp_path)
    remote = {}
    sent, _ = pd.one_round(str(root), FakePusher(), remote, {}, OPTS, "d/S/")
    assert {p for p, _, _ in sent} == {"d/S/graphs/codebase.html", "d/S/runs.json", "d/S/runs/r1/live.json"}
    sent, _ = pd.one_round(str(root), FakePusher(), remote, {}, OPTS, "d/M/")   # same names, another folder
    assert {p for p, _, _ in sent} == {"d/M/graphs/codebase.html", "d/M/runs.json", "d/M/runs/r1/live.json"}
    remote.update({"d/S/runs/gone/live.json": "0c", "runs/old/live.json": "0d", "cluster/squeue.json": "0e"})
    prune = SimpleNamespace(**{**vars(OPTS), "prune": True})
    _, deleted = pd.one_round(str(root), FakePusher(), dict(remote), {}, prune, "d/S/")
    assert deleted == ["d/S/runs/gone/live.json"]               # never d/M/, the top level or cluster/
    _, deleted = pd.one_round(str(root), FakePusher(), dict(remote), {}, prune, "")
    assert deleted == ["runs/old/live.json"]                    # the top level never prunes the folders
