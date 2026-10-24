"""
Benchmark validity: is the benchmark measuring reasoning, and are its inputs faithful?

  audit          surface-cue audit of the counterfactual distractors (hypothesis-only test): a classifier that
                 sees ONLY the two texts (true KU and its edit, no paper context) tries to tell which one is the
                 original (char n-gram TF-IDF, logistic regression, 5-fold cross-validation grouped by KU). An
                 edit it spots with probability >= mp_filter_tau is detectable from wording alone and is
                 removed from evaluation when mp_filter is on (adversarial filtering, as AFLite / HellaSwag).
                 Also reports how often BM25-style word overlap prefers the edit.
  faithfulness   are the LLM-generated KUs supported by the paper's full text (parquet column text_sanitized)?
                 Per KU: share of its content words found in the best-matching 3-sentence window of the paper,
                 share found anywhere, and share of its numbers found in the paper (unsupported numbers =
                 possible hallucination). Sampled papers of the test split.
  export         an expert-annotation sheet (CSV, identical for every annotator, plus a hidden key):
                 counterfactual items (context + two options: which one is the paper's? is the edit fluent and
                 meaning-changing?) and faithfulness items (KU + best-matching passage: faithful?).
  agreement      reads the filled sheets of two or more annotators: human accuracy on the counterfactual items,
                 edit validity, KU faithfulness, and Cohen's kappa between annotators per question.
  contamination  from an experiment's results: every task for test papers up to vs after the text encoders'
                 pretraining cutoff (by_period); a drop after the cutoff suggests memorised test papers.

    python -m sci_jepa.evaluation.validity audit --set max_papers=500
    python -m sci_jepa.evaluation.validity faithfulness --set max_papers=500 --n 200
    python -m sci_jepa.evaluation.validity export --set max_papers=500 --n 200 --out annotation/
    python -m sci_jepa.evaluation.validity agreement annotation/annotator_A.csv annotation/annotator_B.csv
    python -m sci_jepa.evaluation.validity contamination paper_a_results/BENCH
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

from sci_jepa.data import ku_schema as S

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_STOP = set("the a an of and or to in on for with by from at as is are was were be been this that these those it its "
            "we our their which than then also into over under between can may not no more most such using used use "
            "based results result show shows paper study method methods approach model models data".split())


def _validity_path(cfg):
    from sci_jepa.data.ku_graph import data_fingerprint
    return os.path.join(cfg.index_dir, f"validity_{data_fingerprint(cfg)}.json")


# ---------------------------------------------------------------- surface-cue audit + adversarial filter
def audit(gi, cfg, save=True):
    """-> {"by_kind": {kind: {...}}, "flagged": [mp rows], ...}. Saved next to the index (cached)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    texts, mp_texts = gi.texts()
    rows = np.arange(len(gi.mp_comp))
    if len(rows) < 20:
        return {"n": int(len(rows)), "note": "too few counterfactuals for a cross-validated audit", "flagged": []}
    orig = [texts[int(gi.mp_comp[r])] for r in rows]
    edit = [mp_texts[int(r)] for r in rows]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200000)
    vec.fit(orig + edit)
    A, B = vec.transform(orig), vec.transform(edit)
    # pairwise, order-balanced: x = a - b, y = 1 if a is the original; each pair appears in both orders
    X = __import__("scipy.sparse", fromlist=["vstack"]).vstack([A - B, B - A]).tocsr()
    y = np.concatenate([np.ones(len(rows)), np.zeros(len(rows))])
    groups = np.concatenate([gi.mp_comp[rows], gi.mp_comp[rows]])
    p = np.zeros(len(y))
    k = min(5, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    p_orig = (p[:len(rows)] + (1 - p[len(rows):])) / 2       # probability of naming the original, both orders
    tau = cfg.mp_filter_tau
    kind_acc = {int(k): float((p_orig[gi.mp_kind[rows] == k] >= 0.5).mean()) for k in np.unique(gi.mp_kind[rows])}
    bad_kinds = {k for k, acc in kind_acc.items() if acc >= cfg.mp_kind_tau}   # a whole kind is an artifact
    flagged = [int(r) for r, q, k in zip(rows, p_orig, gi.mp_kind[rows]) if q >= tau or int(k) in bad_kinds]
    # word-overlap cue: does BM25-like overlap with the paper's other KUs prefer the edit?
    def overlap(t, ctx):
        a, b = set(w.lower() for w in _WORD.findall(t)) - _STOP, ctx
        return len(a & b) / max(len(a), 1)
    pref_edit = []
    for r in rows:
        c = int(gi.mp_comp[r]); p0 = int(gi.comp_paper[c])
        ctx = set(w.lower() for cc in gi.paper_comps[p0] if int(cc) != c for w in _WORD.findall(texts[int(cc)])) - _STOP
        pref_edit.append(overlap(mp_texts[int(r)], ctx) > overlap(texts[c], ctx))
    by = {}
    for kind_id in np.unique(gi.mp_kind):
        m = gi.mp_kind[rows] == kind_id
        by[S.MP_KINDS[int(kind_id)]] = {"n": int(m.sum()), "detector_accuracy": float((p_orig[m] >= 0.5).mean()),
                                        "flagged": int(m.sum()) if int(kind_id) in bad_kinds else int((p_orig[m] >= tau).sum()),
                                        "removed_as_kind": int(kind_id) in bad_kinds,
                                        "overlap_prefers_edit": float(np.mean(np.asarray(pref_edit)[m]))}
    out = {"n": int(len(rows)), "tau": tau, "detector_accuracy": float((p_orig >= 0.5).mean()),
           "flagged_share": float(len(flagged) / len(rows)), "by_kind": by, "flagged": flagged,
           "kind_tau": cfg.mp_kind_tau,
           "removed_kinds": [S.MP_KINDS[k] for k in sorted(bad_kinds)],
           "note": "pairwise hypothesis-only detector (no paper context); chance 0.5. Removed when mp_filter is on: "
                   "edits it names with probability >= tau, and every edit of a kind it detects in >= kind_tau of "
                   "cases (a systematic wording artifact)."}
    if save and not cfg.synthetic:
        try:
            os.makedirs(cfg.index_dir, exist_ok=True)
            json.dump(out, open(_validity_path(cfg), "w"), indent=1)
        except OSError:                                   # read-only index folder: recomputed next time
            pass
    return out


def apply_filter(gi, cfg, log=print):
    """mp_filter: drop the counterfactuals the audit flags (computed once per index, cached)."""
    if getattr(gi, "_mp_filtered", False) or not len(getattr(gi, "mp_comp", [])):
        return None
    path = _validity_path(cfg)
    res = json.load(open(path)) if os.path.exists(path) and not cfg.synthetic else audit(gi, cfg)
    bad = set(res.get("flagged", []))
    for c in list(gi.mp_of):
        gi.mp_of[c] = [r for r in gi.mp_of[c] if r not in bad]
    gi._mp_filtered = True
    log(f"[validity] adversarial filter: removed {len(bad)} of {res.get('n', 0)} counterfactuals detectable from "
        f"wording alone (pairwise detector accuracy {res.get('detector_accuracy', float('nan')):.2f}, "
        f"tau {res.get('tau')})")
    return res


# ---------------------------------------------------------------- faithfulness to the full text
def _full_texts(cfg, paper_ids):
    import pyarrow.parquet as pq
    from sci_jepa.data.ku_graph import shard_files
    want, out = set(paper_ids), {}
    for f in shard_files(cfg):
        pf = pq.ParquetFile(f)
        if "text_sanitized" not in pf.schema_arrow.names:
            continue
        t = pf.read(columns=["paper_id", "text_sanitized"])
        ids = [str(x) for x in t.column("paper_id").to_pylist()]
        hit = [i for i, x in enumerate(ids) if x in want and x not in out]
        if hit:
            col = t.column("text_sanitized")
            for i in hit:
                out[ids[i]] = col[i].as_py() or ""
        if len(out) == len(want):
            break
    return out


def _support(ku, sents, full_words, full_text):
    w = [x.lower() for x in _WORD.findall(ku) if x.lower() not in _STOP]
    if not w:
        return None
    ws = set(w)
    span = int(min(12, max(3, 2 * len(S._SENT.split(ku)))))   # window grows with the KU: sections summarise paragraphs
    best, best_i = 0.0, 0
    for i in range(len(sents)):
        win = set(x.lower() for x in _WORD.findall(" ".join(sents[i:i + span])))
        sc = len(ws & win) / len(ws)
        if sc > best:
            best, best_i = sc, i
    nums = _NUMBER.findall(ku)
    num_ok = [n in full_text for n in nums]
    return {"window": best, "anywhere": len(ws & full_words) / len(ws),
            "numbers": float(np.mean(num_ok)) if nums else None, "n_numbers": len(nums),
            "passage": " ".join(sents[best_i:best_i + span])[:1500], "window_sentences": span}


def faithfulness(gi, cfg, n=200, seed=0, split="test"):
    texts, _ = gi.texts()
    rng = np.random.default_rng(seed)
    papers = [int(p) for p in rng.permutation(gi.papers(split))[:n]]
    full = _full_texts(cfg, [str(gi.paper_ids[p]) for p in papers])
    rows = []
    for p in papers:
        ft = full.get(str(gi.paper_ids[p]))
        if not ft:
            continue
        sents = [x for x in S._SENT.split(ft) if len(x) > 20]
        fw = set(x.lower() for x in _WORD.findall(ft))
        for c in gi.paper_comps[p]:
            r = _support(texts[int(c)], sents, fw, ft)
            if r:
                rows.append({"comp": int(c), "paper": str(gi.paper_ids[p]), "type": S.COMP_TYPES[gi.comp_type[c]], **r})
    if not rows:
        return {"n": 0, "note": "no full texts found (text_sanitized) for the sampled papers"}
    by = {}
    for t in sorted({r["type"] for r in rows}):
        g = [r for r in rows if r["type"] == t]
        nums = [r["numbers"] for r in g if r["numbers"] is not None]
        by[t] = {"n": len(g), "window_support": float(np.mean([r["window"] for r in g])),
                 "anywhere_support": float(np.mean([r["anywhere"] for r in g])),
                 "low_support_share": float(np.mean([r["window"] < 0.5 for r in g])),
                 "numbers_found": float(np.mean(nums)) if nums else None}
    nums = [r["numbers"] for r in rows if r["numbers"] is not None]
    return {"papers": len(full), "n": len(rows), "window_support": float(np.mean([r["window"] for r in rows])),
            "anywhere_support": float(np.mean([r["anywhere"] for r in rows])),
            "low_support_share": float(np.mean([r["window"] < 0.5 for r in rows])),
            "numbers_found": float(np.mean(nums)) if nums else None, "by_type": by,
            "lowest": sorted(rows, key=lambda r: r["window"])[:25],
            "note": "content-word overlap of each KU with its best window of the paper's full text (3-12 sentences, "
                    "twice the KU's sentences) and with the whole paper; low window support (< 0.5) or numbers not in "
                    "the paper mark KUs to check by hand"}


# ---------------------------------------------------------------- expert annotation
_TASK_OF = {"result": "forward", "claim": "abduction", "method": "inverse", "procedure": "inverse"}


def export(gi, cfg, out_dir, n=200, annotators=("A", "B"), seed=0):
    """Write annotation sheets: n counterfactual items + n // 2 faithfulness items."""
    texts, mp_texts = gi.texts()
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    items, key = [], []
    test = set(int(p) for p in gi.papers("test"))
    rows = [int(r) for r in rng.permutation(len(gi.mp_comp)) if int(gi.comp_paper[gi.mp_comp[r]]) in test][:n]
    for k, r in enumerate(rows):
        c = int(gi.mp_comp[r]); p = int(gi.comp_paper[c]); t = S.COMP_TYPES[gi.comp_type[c]]
        given = S.TEMPLATES[_TASK_OF.get(t, "forward")][0]
        ctx = "\n".join(f"{S._SECTION_TITLE.get(S.COMP_TYPES[gi.comp_type[x]], S.COMP_TYPES[gi.comp_type[x]])}: "
                        f"{texts[int(x)]}" for x in gi.paper_comps[p] if S.COMP_TYPES[gi.comp_type[x]] in given)
        swap = bool(rng.random() < 0.5)
        a, b = (mp_texts[r], texts[c]) if swap else (texts[c], mp_texts[r])
        iid = f"cf{k:04d}"
        items.append({"item_id": iid, "kind": "counterfactual", "task": _TASK_OF.get(t, ""), "hidden_part": t,
                      "context": ctx, "option_A": a, "option_B": b,
                      "Q1_which_is_the_papers_A_or_B": "", "Q2_edited_option_is_fluent_1to5": "",
                      "Q3_edit_changes_the_meaning_yes_no": "", "comment": ""})
        key.append({"item_id": iid, "original": "B" if swap else "A", "edit_kind": S.MP_KINDS[int(gi.mp_kind[r])],
                    "mp_row": r})
    fa = faithfulness(gi, cfg, n=max(n // 10, 20), seed=seed)
    pool = fa.get("lowest", []) if fa.get("n") else []
    for k, r in enumerate(pool[:n // 2]):
        iid = f"fa{k:04d}"
        items.append({"item_id": iid, "kind": "faithfulness", "task": "", "hidden_part": r["type"],
                      "context": r["passage"], "option_A": texts[r["comp"]], "option_B": "",
                      "Q1_which_is_the_papers_A_or_B": "n/a", "Q2_edited_option_is_fluent_1to5": "n/a",
                      "Q3_edit_changes_the_meaning_yes_no": "n/a", "Q4_KU_is_faithful_to_the_paper_yes_partly_no": "",
                      "comment": ""})
        key.append({"item_id": iid, "comp": r["comp"], "paper": r["paper"], "window_support": r["window"]})
    order = list(rng.permutation(len(items)))
    cols = ["item_id", "kind", "task", "hidden_part", "context", "option_A", "option_B",
            "Q1_which_is_the_papers_A_or_B", "Q2_edited_option_is_fluent_1to5", "Q3_edit_changes_the_meaning_yes_no",
            "Q4_KU_is_faithful_to_the_paper_yes_partly_no", "comment"]
    for a in annotators:
        with open(os.path.join(out_dir, f"annotator_{a}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for i in order:
                w.writerow({c: _cell(items[i].get(c, "")) for c in cols})
    json.dump(key, open(os.path.join(out_dir, "key_do_not_share.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "INSTRUCTIONS.txt"), "w") as f:
        f.write("For every counterfactual row: read the context (the visible parts of a paper). Q1: which option is the "
                "paper's real text (A or B)? Q2: how fluent and plausible is the OTHER option (1 = broken, 5 = could be "
                "real)? Q3: does it change the meaning so that only the real text fits (yes / no)?\nFor every "
                "faithfulness row: the context is the best-matching passage of the full paper. Q4: is the knowledge unit "
                "(option_A) faithful to the paper (yes / partly / no)?\nWork independently; do not discuss items.\n")
    return {"items": len(items), "counterfactual": len(rows), "faithfulness": min(len(pool), n // 2), "dir": out_dir}


def _cell(v):
    """CSV cell without NUL bytes (some extracted paper texts contain them; the csv module cannot write them)."""
    return v.replace("\x00", "") if isinstance(v, str) else v


def _kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return None
    po = float((a == b).mean())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)
    return float((po - pe) / max(1 - pe, 1e-9))


def agreement(files, key_path=None):
    sheets = {}
    for fp in files:
        with open(fp) as f:
            sheets[os.path.basename(fp)] = {r["item_id"]: r for r in csv.DictReader(f)}
    key_path = key_path or os.path.join(os.path.dirname(files[0]), "key_do_not_share.json")
    key = {k["item_id"]: k for k in json.load(open(key_path))}
    norm = lambda v: (v or "").strip().lower()
    out = {"annotators": list(sheets), "per_annotator": {}, "kappa": {}}
    for name, s in sheets.items():
        cf = [r for r in s.values() if r["kind"] == "counterfactual" and norm(r["Q1_which_is_the_papers_A_or_B"])]
        fa = [r for r in s.values() if r["kind"] == "faithfulness" and norm(r.get("Q4_KU_is_faithful_to_the_paper_yes_partly_no"))]
        flu = [float(r["Q2_edited_option_is_fluent_1to5"]) for r in cf if norm(r["Q2_edited_option_is_fluent_1to5"]).replace(".", "").isdigit()]
        out["per_annotator"][name] = {
            "human_accuracy": float(np.mean([norm(r["Q1_which_is_the_papers_A_or_B"]).upper()[:1] == key[r["item_id"]]["original"]
                                             for r in cf])) if cf else None,
            "edit_fluency_mean": float(np.mean(flu)) if flu else None,
            "edit_changes_meaning": float(np.mean([norm(r["Q3_edit_changes_the_meaning_yes_no"]).startswith("y") for r in cf])) if cf else None,
            "ku_faithful": float(np.mean([norm(r["Q4_KU_is_faithful_to_the_paper_yes_partly_no"]) == "yes" for r in fa])) if fa else None,
            "ku_faithful_or_partly": float(np.mean([norm(r["Q4_KU_is_faithful_to_the_paper_yes_partly_no"]) in ("yes", "partly") for r in fa])) if fa else None,
            "n_counterfactual": len(cf), "n_faithfulness": len(fa)}
    names = list(sheets)
    for q in ("Q1_which_is_the_papers_A_or_B", "Q3_edit_changes_the_meaning_yes_no", "Q4_KU_is_faithful_to_the_paper_yes_partly_no"):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = sheets[names[i]], sheets[names[j]]
                common = [k for k in A if k in B and norm(A[k].get(q)) not in ("", "n/a") and norm(B[k].get(q)) not in ("", "n/a")]
                if common:
                    out["kappa"][f"{q} {names[i]}~{names[j]}"] = {"kappa": _kappa([norm(A[k][q]) for k in common],
                                                                                  [norm(B[k][q]) for k in common]),
                                                                  "n": len(common)}
    return out


# ---------------------------------------------------------------- contamination
def contamination(exp_dir, min_n=50):
    """Per system and task: score before vs after the encoders' cutoff, and the drop -- only where both periods have
    >= min_n test queries (a handful of after-cutoff papers says nothing)."""
    rows = {}
    for f in glob.glob(os.path.join(exp_dir, "*_seed*.json")):
        m = json.load(open(f))
        T = m.get("tasks") or {}
        rec = rows.setdefault(m.get("tag"), {})
        for k in ("forward", "inverse", "abduction"):
            bp = (T.get(k) or {}).get("by_period") or {}
            for per, v in bp.items():
                rec.setdefault(f"{k}:{per}", []).append(v["ratio"])
                rec.setdefault(f"{k}:{per}:n", []).append(v.get("n", 0))
        bp = (T.get("consistency") or {}).get("by_period") or {}
        for per, v in bp.items():
            rec.setdefault(f"consistency:{per}", []).append(v["bal_acc"])
    out = {}
    for tag, rec in rows.items():
        r = {k: float(np.mean(v)) for k, v in rec.items() if not k.endswith(":n")}
        for k in ("forward", "inverse", "abduction", "consistency"):
            a, b = r.get(f"{k}:until_cutoff"), r.get(f"{k}:after_cutoff")
            na, nb = sum(rec.get(f"{k}:until_cutoff:n", [0])), sum(rec.get(f"{k}:after_cutoff:n", [0]))
            r[f"{k}:n_after"] = nb
            if a is not None and b is not None and na >= min_n and nb >= min_n:
                r[f"{k}:drop_after_cutoff"] = a - b
        out[tag] = r
    return out


# ---------------------------------------------------------------- CLI
def main():
    from sci_jepa.config import PaperAConfig
    from sci_jepa.data.graph_index import get_index
    from sci_jepa.training.train import parse_overrides
    ap = argparse.ArgumentParser("validity")
    ap.add_argument("cmd", choices=["audit", "faithfulness", "export", "agreement", "contamination"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="annotation")
    a = ap.parse_args()
    if a.cmd == "agreement":
        print(json.dumps(agreement(a.paths), indent=1)); return
    if a.cmd == "contamination":
        print(json.dumps(contamination(a.paths[0]), indent=1)); return
    cfg = PaperAConfig().override(**parse_overrides(a.set))
    gi = get_index(cfg)
    if a.cmd == "audit":
        r = audit(gi, cfg); r.pop("flagged", None); print(json.dumps(r, indent=1))
    elif a.cmd == "faithfulness":
        r = faithfulness(gi, cfg, n=a.n); r.pop("lowest", None); print(json.dumps(r, indent=1))
    else:
        print(json.dumps(export(gi, cfg, a.out, n=a.n), indent=1))


if __name__ == "__main__":
    main()
