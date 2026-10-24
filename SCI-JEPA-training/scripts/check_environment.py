#!/usr/bin/env python
"""Environment check before a Paper A run: GPU, real data, text encoder, dashboard tunnel.

Uses no SCI-JEPA model code, so it separates setup problems from code problems.
    python scripts/check_environment.py                     # inside a GPU job / on a GPU machine
    bash cluster/grete/run_paper_a.sh submit check          # on Grete: submits it and opens the page
  1. torch / CUDA / GPU
  2. reads N_CHECK papers (default 1000) from the normalized parquet, parses their KUs
  3. embeds the KUs with the Paper A text encoder (gte-modernbert-base) on the GPU, fp16
  4. a tiny real-data retrieval task: from a paper's key result, find its own research question
     among all N_CHECK papers (MRR vs chance), overall and per field
  5. serves a status page from its own folder and writes server_<job>.json into the results' dashboard
     folder (never its index.html), so `run_paper_a.sh tunnel`
     opens it in the browser; stays up KEEP_SERVING minutes (default 10)
"""
import glob
import html
import http.server
import json
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

PARQUET = os.environ["PAPER_A_PARQUET"]
OUT = Path(os.environ["PAPER_A_OUT"]) / "server_check"
DASH = OUT / "dashboard"                                   # its own page: never touches the results' dashboard
MAIN_DASH = Path(os.environ["PAPER_A_OUT"]) / "dashboard"  # only server_<job>.json goes here, for the tunnel
N = int(os.environ.get("N_CHECK", 1000))
KEEP = int(os.environ.get("KEEP_SERVING", 10))
KU_FIELDS = ["research_context", "research_question_hypothesis", "methodological_details",
             "procedures_architectures", "key_results", "interpretation_implications"]
OUT.mkdir(parents=True, exist_ok=True); DASH.mkdir(parents=True, exist_ok=True)
STATUS = {"steps": []}


def step(name, **info):
    STATUS["steps"].append({"step": name, "time": time.strftime("%H:%M:%S"), **info})
    print(f"[{STATUS['steps'][-1]['time']}] {name}: {json.dumps(info, default=str)}", flush=True)
    page()


def page():
    rows = "".join(f"<tr><td>{html.escape(s['step'])}</td><td>{s['time']}</td><td><pre>"
                   f"{html.escape(json.dumps({k: v for k, v in s.items() if k not in ('step', 'time')}, indent=1, default=str))}"
                   f"</pre></td></tr>" for s in STATUS["steps"])
    (DASH / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>Paper A server check</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1000px}}td,th{{border-bottom:1px solid #ddd;padding:6px;vertical-align:top}}
pre{{margin:0;white-space:pre-wrap}}</style>
<h1>Paper A — server check</h1><p>job {os.environ.get('SLURM_JOB_ID')} on {socket.gethostname()} · refreshes every 5 s</p>
<table><tr><th>step</th><th>time</th><th>result</th></tr>{rows}</table>""")


# 5 first: the dashboard is up before the work starts, as in paper_a_run.py
port = next(p for p in range(8765, 8865) if socket.socket().connect_ex(("127.0.0.1", p)) != 0)
handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(DASH), **k)
http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
MAIN_DASH.mkdir(parents=True, exist_ok=True)
for n in (f"server_{os.environ.get('SLURM_JOB_ID')}.json",):
    (MAIN_DASH / n).write_text(json.dumps({"node": socket.gethostname(), "port": port,
                                      "job": os.environ.get("SLURM_JOB_ID"), "check": True}, indent=1))
step("dashboard", node=socket.gethostname(), port=port)

# 1. GPU
step("gpu", torch=torch.__version__, cuda_build=torch.version.cuda, cuda=torch.cuda.is_available(),
     device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
     memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "no CUDA device"

# 2. real data
t0 = time.time()
files = sorted(glob.glob(os.path.join(PARQUET, "*.parquet")))
papers = []
for f in files[:: max(len(files) // 20, 1)]:              # spread over the shards: different years / fields
    t = pq.ParquetFile(f).read_row_group(0, columns=["paper_id", "field_subfield", "claims"] + KU_FIELDS).to_pylist()
    papers += [r for r in t if r["key_results"] and r["research_question_hypothesis"]][: N // 20 + 1]
    if len(papers) >= N:
        break
papers = papers[:N]
n_claims = sum(len(json.loads(p["claims"])) if p["claims"] else 0 for p in papers)
fields = [(p["field_subfield"] or "?").split("—")[0].strip() for p in papers]
step("data", shards=len(files), papers=len(papers), claims=n_claims, sec=round(time.time() - t0, 1),
     top_fields={k: int(v) for k, v in sorted(zip(*np.unique(fields, return_counts=True)), key=lambda x: -x[1])[:8]})

# 3. text encoder on the GPU
from sentence_transformers import SentenceTransformer
t0 = time.time()
enc = SentenceTransformer("Alibaba-NLP/gte-modernbert-base", device="cuda", model_kwargs={"torch_dtype": torch.float16})
texts = [p[k] or "" for p in papers for k in KU_FIELDS]
E = enc.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
E = E.reshape(len(papers), len(KU_FIELDS), -1)
sec = time.time() - t0
step("embedding", model="gte-modernbert-base fp16", kus=len(texts), dim=E.shape[-1], sec=round(sec, 1),
     kus_per_sec=round(len(texts) / sec), peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))

# 4. key result -> own research question, among all papers
q, d = E[:, KU_FIELDS.index("key_results")], E[:, KU_FIELDS.index("research_question_hypothesis")]
S = q @ d.T
rank = (S > S[np.arange(len(S)), np.arange(len(S))][:, None]).sum(1) + 1
mrr = float(np.mean(1 / rank)); chance = float(np.mean(1 / np.arange(1, len(S) + 1)))
by_field = {}
for f in set(fields):
    idx = [i for i, x in enumerate(fields) if x == f]
    if len(idx) >= 20:
        by_field[f] = {"papers": len(idx), "MRR": round(float(np.mean(1 / rank[idx])), 3)}
res = {"papers": len(S), "MRR": round(mrr, 4), "chance_MRR": round(chance, 4), "x_chance": round(mrr / chance, 1),
       "hits@1": round(float(np.mean(rank == 1)), 3), "hits@10": round(float(np.mean(rank <= 10)), 3), "by_field": by_field}
step("retrieval: key result -> own research question", **res)
(OUT / "result.json").write_text(json.dumps(STATUS, indent=1, default=str))
print("written:", OUT / "result.json", flush=True)

if KEEP > 0:
    step("done", note=f"dashboard stays up {KEEP} min")
    time.sleep(60 * KEEP)
srv.shutdown()
