"""
Before / after accuracy check of the scaling changes (scaling/README.md): the same arms, seeds, papers and epochs,
trained with the code before the changes (scaling/code_before, PYTHONPATH) or the current code with the new speed
settings. Writes <out>/<arm>_seed<s>.json (the usual result files) and <out>/check.json (accuracy + wall time).

    sbatch scaling/accuracy_check.sbatch before        # or: after
"""
import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--split", required=True)
p.add_argument("--out", required=True)
p.add_argument("--variant", choices=["before", "after"], required=True)
p.add_argument("--n-papers", type=int, default=500)
p.add_argument("--epochs", type=float, default=10)
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--arms", default="reference,jepa+rgat")
p.add_argument("--workers", type=int, default=7, help="after: batch-building processes")
A = p.parse_args()
os.makedirs(A.out, exist_ok=True)
os.environ.setdefault("PAPER_A_INDEX_DIR", os.path.join(A.out, "index"))

import sci_jepa  # noqa: E402
from sci_jepa import config as C  # noqa: E402
from sci_jepa.data.graph_index import get_index  # noqa: E402
from sci_jepa.training.train import run_experiment  # noqa: E402
from sci_jepa.evaluation.tasks import reasoning_index  # noqa: E402

speed = dict(C.FAST) if A.variant == "before" else dict(C.FAST_GPU, batch_workers=A.workers)
cfg = C.PaperAConfig().override(**C.SIZES["local"], max_papers=A.n_papers, split_dir=A.split, epochs=A.epochs,
                                patience=0, seeds=[int(s) for s in A.seeds.split(",")], device="auto", batch_size=64,
                                rows_per_shard=100, out_dir=A.out, model_dir=os.path.join(A.out, "models"),
                                index_dir=os.environ["PAPER_A_INDEX_DIR"], **speed)
print("code:", os.path.dirname(sci_jepa.__file__), "| variant:", A.variant, "| speed:", speed, flush=True)
gi = get_index(cfg)
bench = dict(C.EXPERIMENTS["BENCH"])
arms = [(a, bench[a]) for a in A.arms.split(",")]
t0 = time.time()
res, _ = run_experiment(cfg, gi, arms=arms, name="check", log=lambda *a: print(*a, flush=True))
rows = []
for tag, runs in res.items():
    for m in runs:
        ri = reasoning_index(m["tasks"], cfg.primary_average) if m.get("tasks") else None
        rows.append({"arm": tag, "seed": m["seed"], "bal_MRR": m["retrieval_balanced"]["MRR"],
                     "bal_chance": m["retrieval_balanced"].get("chance_MRR"),
                     "reasoning_index": ri if not isinstance(ri, dict) else ri.get("index", ri),
                     "train_sec": m.get("train_sec"), "best_epoch": (m.get("run_length") or {}).get("best_epoch")})
json.dump({"variant": A.variant, "speed": speed, "papers": A.n_papers, "epochs": A.epochs, "total_sec": time.time() - t0,
           "runs": rows}, open(os.path.join(A.out, "check.json"), "w"), indent=1, default=str)
for r in rows:
    print(r, flush=True)
print(f"total {time.time() - t0:.0f}s", flush=True)
