"""
Stored, domain-balanced train / validation / test split of the normalized Scientific-Summaries (run once).

    sci-jepa split --parquet data/normalized/arxiv,data/normalized/pubmed --out data/splits/d0 \\
                   --ku-fixes data/raw-KU-Fixes --train-papers 60000 --val-papers 5000 --test-papers 5000

Training and benchmarking then read the stored split (PAPER_A_SPLIT=data/splits/d0, or --split-dir) and never
sample or split again.

How the split is made
  1. metadata pass over every paper: publication year (ku_schema.paper_year: source_year, oa_year, or the arXiv
     id), canonical research domain (ku_schema.domain_of) and subset; only cheap columns are read
  2. split by year: train <= --train-until (2023), validation = --val-years (2024), test = --test-years (2025 2026).
     Papers without a year, or from other years, are left out and counted in the manifest
  3. cleaning with the KU-Fixes index (--ku-fixes): every copy of a duplicated paper except the one to keep is
     dropped, so no paper can sit in two splits; near-empty summaries (needs_regeneration) are dropped
  4. domain balance: arXiv / PubMed are dominated by a few domains, and differently per year (Computer Science is
     17% of the train years but 47% of the test years). One set of domain shares is computed from the pooled
     counts of all three splits by temperature sampling (share ~ count^alpha: alpha 0.5 = square root, 0 = equal,
     1 = natural; the same rule as ku_graph.select_balanced) and applied to EVERY split, so train, validation and
     test have the same domain mix. A domain with too few papers in a split gives its leftover to the others
  5. quality: candidates are checked with the index build's own KU rules (ku_schema.record_to_kus,
     min_kus_per_paper, required_kus), so every stored paper is used; a rejected one is replaced by the next
     paper of the same split and domain
  6. output: <out>/{train,val,test}/part-*.parquet (the normalized schema plus _split and _split_rank, a random
     order inside the split), papers.csv (one row per paper), manifest.json (settings, counts at every step)
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from sci_jepa.data import ku_schema as S

SPLITS = ("train", "val", "test")
META_COLS = ["paper_id", "subset", "source_year", "oa_year", "field_subfield"]
MANIFEST = "manifest.json"
FORMAT_VERSION = 1


# ---------------------------------------------------------------- reading
def parquet_files(dirs):
    files = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if not found:
            sys.exit(f"no parquet files in {d}")
        files += found
    return files


def read_rows(path, rows, columns):
    """Rows `rows` (sorted ints) of a parquet file, reading only the row groups that hold them. -> list of dicts."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = [c for c in columns if c in pf.schema_arrow.names]
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    rows = np.asarray(rows)
    groups = np.unique(np.searchsorted(starts, rows, side="right") - 1)
    t = pa.concat_tables([pf.read_row_group(int(g), columns=cols) for g in groups])
    offset = np.concatenate([[0], np.cumsum([starts[g + 1] - starts[g] for g in groups])])
    base = {int(g): int(o) for g, o in zip(groups, offset)}
    local = [base[int(g)] + int(r - starts[g]) for r, g in zip(rows, np.searchsorted(starts, rows, side="right") - 1)]
    return t.take(np.asarray(local)).to_pylist()


def ku_fix_drops(ku_fixes, duplicates):
    """{(folder, file name, row)} to drop: duplicate copies (duplicates.parquet) and near-empty summaries."""
    import pyarrow.parquet as pq
    drop, n_dup, n_regen = set(), 0, 0
    if not ku_fixes:
        return drop, {"duplicates": 0, "needs_regeneration": 0, "ku_fixes": None}

    def key(file, row):                               # "data/arxiv/new_00200.parquet" -> ("arxiv", "new_00200.parquet")
        parts = file.split("/")
        return parts[-2], parts[-1], int(row)

    if duplicates != "none":
        d = pq.read_table(os.path.join(ku_fixes, "duplicates.parquet"), columns=["file", "row_index", "keep", "keep_strong"])
        flag = "keep" if duplicates == "all" else "keep_strong"
        for f, r, k in zip(d["file"].to_pylist(), d["row_index"].to_pylist(), d[flag].to_pylist()):
            if not k:
                drop.add(key(f, r)); n_dup += 1
    fx = pq.read_table(os.path.join(ku_fixes, "ku_fix_index.parquet"), columns=["file", "row_index", "needs_regeneration"])
    for f, r, n in zip(fx["file"].to_pylist(), fx["row_index"].to_pylist(), fx["needs_regeneration"].to_pylist()):
        if n:
            drop.add(key(f, r)); n_regen += 1
    return drop, {"duplicates": n_dup, "needs_regeneration": n_regen, "ku_fixes": os.path.abspath(ku_fixes),
                  "duplicate_tier": duplicates}


# ---------------------------------------------------------------- balance
def balanced_quota(weights, avail, total):
    """Papers per domain: shares proportional to `weights`, never more than a domain has (`avail`); what a domain
    cannot fill goes to the domains that still have papers (as ku_graph._temperature_quota, with the weights
    given separately so every split can use the same pooled shares). total <= 0: everything available."""
    avail = {d: n for d, n in avail.items() if n > 0}
    if total <= 0:
        return dict(avail)
    quota, left = {d: 0 for d in avail}, int(min(total, sum(avail.values())))
    while left > 0:
        open_ = [d for d in avail if quota[d] < avail[d]]
        if not open_:
            break
        w = np.asarray([weights.get(d, 0.0) for d in open_], float)
        w = np.ones(len(open_)) if w.sum() <= 0 else w
        w /= w.sum()
        add = np.floor(w * left).astype(int)
        add[np.argsort(-w)[:left - add.sum()]] += 1
        moved = 0
        for d, a in zip(open_, add):
            g = int(min(a, avail[d] - quota[d])); quota[d] += g; moved += g
        left -= moved
        if moved == 0:
            break
    return quota


def domain_weights(pooled, mode, alpha):
    a = {"temperature": alpha, "uniform": 0.0, "natural": 1.0}[mode]
    return {d: float(n) ** a for d, n in pooled.items() if n > 0}, a


# ---------------------------------------------------------------- the split
def make_split(a):
    from sci_jepa.config import PaperAConfig
    cfg = PaperAConfig()
    t0 = time.time()
    dirs = [d for d in a.parquet.split(",") if d]
    files = parquet_files(dirs)
    folder = [os.path.basename(os.path.dirname(f).rstrip("/")) for f in files]
    year_split = {y: "val" for y in a.val_years}
    year_split.update({y: "test" for y in a.test_years})
    drop, cleaning = ku_fix_drops(a.ku_fixes, a.duplicates)
    rng = np.random.default_rng(a.seed)

    # 1-3: metadata, year split, cleaning
    import pyarrow.parquet as pq
    cand = {s: defaultdict(list) for s in SPLITS}          # split -> domain -> [(file idx, row)]
    excluded, years_seen = Counter(), {s: Counter() for s in SPLITS}
    subsets = set(a.subsets or [])
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=META_COLS).to_pylist()
        name = os.path.basename(f)
        for r, rec in enumerate(t):
            if subsets and rec.get("subset") not in subsets:
                excluded["other_subset"] += 1; continue
            y = S.paper_year(rec)
            if y is None:
                excluded["no_year"] += 1; continue
            s = "train" if y <= a.train_until else year_split.get(y)
            if s is None:
                excluded["other_years"] += 1; continue
            if (folder[fi], name, r) in drop:
                excluded["duplicate_or_near_empty"] += 1; continue
            cand[s][S.domain_of(rec.get("field_subfield"))].append((fi, r))
            years_seen[s][y] += 1
        if fi % 500 == 0:
            print(f"[split] metadata {fi + 1}/{len(files)} files ({time.time() - t0:.0f}s)", flush=True)
    pool = {s: {d: len(v) for d, v in cand[s].items()} for s in SPLITS}
    pooled = Counter()
    for s in SPLITS:
        pooled.update(pool[s])

    # 4: one set of domain shares for every split
    target = {"train": a.train_papers, "val": a.val_papers, "test": a.test_papers}
    if a.balance == "none":
        weights, alpha = None, None
    else:
        weights, alpha = domain_weights(pooled, a.balance, a.alpha)
    quota = {}
    for s in SPLITS:
        w = weights if weights is not None else {d: float(n) for d, n in pool[s].items()}   # none: the split's own mix
        quota[s] = balanced_quota(w, pool[s], target[s])

    # 5: quality check with the index build's KU rules; rejected papers are replaced from the same split and domain
    from sci_jepa.data.ku_graph import READ_COLS
    ku_cols = [c for c in READ_COLS if c not in ("openalex_id", "oa_referenced_works")]

    def usable(rec):
        kus, _, _ = S.record_to_kus(rec, cfg.ku_min_chars, cfg.optional_kus)
        types = {t for t, _, _ in kus}
        return len(kus) >= cfg.min_kus_per_paper and all(t in types for t in cfg.required_kus)

    # every (split, domain) walks its candidates in a random order; each round reads every touched file once
    state = {(s, d): {"order": [cand[s][d][i] for i in rng.permutation(len(cand[s][d]))], "pos": 0, "got": [], "q": q}
             for s in SPLITS for d, q in quota[s].items()}
    rejected, rnd = Counter(), 0
    while True:
        takes = {}
        for key, st in state.items():
            need = st["q"] - len(st["got"])
            if need > 0 and st["pos"] < len(st["order"]):
                takes[key] = st["order"][st["pos"]:st["pos"] + max(int(need * 1.3), 8)]
                st["pos"] += len(takes[key])
        if not takes:
            break
        by_file = defaultdict(set)
        for take in takes.values():
            for fi, r in take:
                by_file[fi].add(r)
        ok = set()
        for fi, rows in by_file.items():
            rows = sorted(rows)
            for r, rec in zip(rows, read_rows(files[fi], rows, ku_cols)):
                if usable(rec):
                    ok.add((fi, r))
        for (s, d), take in takes.items():
            st = state[(s, d)]
            for c in take:                             # keep the random order
                if len(st["got"]) >= st["q"]:
                    break
                if c in ok:
                    st["got"].append(c)
                else:
                    rejected[s] += 1
        rnd += 1
        print(f"[split] quality round {rnd}: {sum(len(t) for t in takes.values())} candidates read from "
              f"{len(by_file)} files ({time.time() - t0:.0f}s)", flush=True)
    chosen = {s: [] for s in SPLITS}
    for (s, d), st in state.items():
        chosen[s] += [(fi, r, d) for fi, r in st["got"]]
        if len(st["got"]) < st["q"]:
            print(f"[split] {s} / {d}: only {len(st['got'])} usable of quota {st['q']}", flush=True)
    for s in SPLITS:
        print(f"[split] {s}: {len(chosen[s])} papers chosen", flush=True)

    # 6: write every split (each source row group is read once), with a random order inside each split
    write_split(a, files, chosen, rng)
    summary = {s: Counter(d for _, _, d in chosen[s]) for s in SPLITS}
    split_id = hashlib.sha1(repr(sorted((s, files[fi], r) for s in SPLITS for fi, r, _ in chosen[s])).encode()).hexdigest()[:12]
    manifest = {
        "format_version": FORMAT_VERSION, "split_id": split_id,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "years": {"train": f"<= {a.train_until}", "val": a.val_years, "test": a.test_years},
        "sources": {d: len(glob.glob(os.path.join(d, "*.parquet"))) for d in dirs},
        "settings": {k: v for k, v in vars(a).items() if k != "func"},
        "quality_rules": {"ku_min_chars": cfg.ku_min_chars, "min_kus_per_paper": cfg.min_kus_per_paper,
                          "required_kus": list(cfg.required_kus), "optional_kus": list(cfg.optional_kus)},
        "excluded": dict(excluded), "cleaning": cleaning,
        "pool_by_domain": {s: dict(sorted(pool[s].items(), key=lambda kv: -kv[1])) for s in SPLITS},
        "years_in_pool": {s: dict(sorted(years_seen[s].items())) for s in SPLITS},
        "balance": {"mode": a.balance, "alpha": alpha,
                    "shares": ({d: round(w / sum(weights.values()), 4) for d, w in sorted(weights.items(), key=lambda kv: -kv[1])}
                               if weights else None),
                    "quota": quota},
        "rejected_by_quality": dict(rejected),
        "selected": {s: {"total": len(chosen[s]), "by_domain": dict(summary[s].most_common())} for s in SPLITS},
        "elapsed_sec": round(time.time() - t0),
    }
    json.dump(manifest, open(os.path.join(a.out, MANIFEST), "w"), indent=1, default=str)
    report(manifest)
    print(f"[split] wrote {a.out} ({time.time() - t0:.0f}s). Use it with: PAPER_A_SPLIT={a.out}")


def unified_schema(files):
    """One schema for every source file. Normalization unifies columns within a folder, not across subsets
    (arXiv stores the OpenAlex columns as text, PubMed as numbers / booleans): a column whose type differs
    between files becomes text. -> (schema, names of the columns turned into text)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    types, order = defaultdict(set), []
    for f in files:
        for fld in pq.ParquetFile(f).schema_arrow:
            if fld.name not in types:
                order.append(fld.name)
            types[fld.name].add(fld.type)
    fields, to_text = [], set()
    for name in order:
        ts = {t for t in types[name] if not pa.types.is_null(t)}
        strings = {t for t in ts if pa.types.is_string(t) or pa.types.is_large_string(t)}
        if len(ts) <= 1:
            fields.append(pa.field(name, next(iter(ts)) if ts else pa.string()))
        elif ts == strings:
            fields.append(pa.field(name, pa.large_string()))
        else:
            fields.append(pa.field(name, pa.large_string())); to_text.add(name)
    return pa.schema(fields), to_text


def _as_text(v):
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, float) and v.is_integer():
        return str(int(v))                             # 2017.0 -> "2017", as the text columns store it
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def write_split(a, files, chosen, rng):
    import pyarrow as pa
    import pyarrow.parquet as pq
    base, to_text = unified_schema(sorted({files[fi] for s in SPLITS for fi, _, _ in chosen[s]}))
    schema = pa.schema(list(base) + [pa.field("_split", pa.string()), pa.field("_split_rank", pa.int32())])
    rank = {}
    for s in SPLITS:
        for k, i in enumerate(rng.permutation(len(chosen[s]))):
            fi, r, _ = chosen[s][i]
            rank[(fi, r)] = (s, k)
    os.makedirs(a.out, exist_ok=True)
    buf, part, writer = {s: [] for s in SPLITS}, Counter(), {}

    def flush(s, final=False):
        while buf[s] and (final or len(buf[s]) >= a.rows_per_part):
            rows, buf[s] = buf[s][:a.rows_per_part], buf[s][a.rows_per_part:]
            d = os.path.join(a.out, s); os.makedirs(d, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), os.path.join(d, f"part-{part[s]:05d}.parquet"),
                           compression="zstd", row_group_size=256)
            part[s] += 1

    by_file = defaultdict(list)
    for (fi, r) in rank:
        by_file[fi].append(r)
    meta_rows = []
    for n, fi in enumerate(sorted(by_file)):
        rows = sorted(by_file[fi])
        for r, rec in zip(rows, read_rows(files[fi], rows, base.names)):
            s, k = rank[(fi, r)]
            for c in to_text:
                rec[c] = _as_text(rec.get(c))
            rec["_split"], rec["_split_rank"] = s, k
            buf[s].append(rec)
            meta_rows.append((rec.get("paper_id"), s, k, S.paper_year(rec), S.domain_of(rec.get("field_subfield")),
                              rec.get("subset"), files[fi], r))
        for s in SPLITS:
            flush(s)
        if n % 250 == 0:
            print(f"[split] writing: {n + 1}/{len(by_file)} source files", flush=True)
    for s in SPLITS:
        flush(s, final=True)
    with open(os.path.join(a.out, "papers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "split", "rank", "year", "domain", "subset", "source_file", "source_row"])
        w.writerows(sorted(meta_rows, key=lambda m: (SPLITS.index(m[1]), m[2])))


def report(m):
    doms = list(dict.fromkeys(d for s in SPLITS for d in m["pool_by_domain"][s]))
    print(f"\n{'domain':32s}" + "".join(f"{s + ' pool':>13s}{s:>8s}" for s in SPLITS))
    for d in doms:
        print(f"{d:32s}" + "".join(f"{m['pool_by_domain'][s].get(d, 0):>13,}{m['selected'][s]['by_domain'].get(d, 0):>8,}"
                                   for s in SPLITS))
    print(f"{'total':32s}" + "".join(f"{sum(m['pool_by_domain'][s].values()):>13,}{m['selected'][s]['total']:>8,}" for s in SPLITS))
    print("excluded:", m["excluded"], "| rejected by KU rules:", m["rejected_by_quality"])


# ---------------------------------------------------------------- reading a stored split (used by ku_graph)
def load_manifest(split_dir):
    p = os.path.join(split_dir, MANIFEST)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found: create the split first (sci-jepa split ... --out {split_dir})")
    return json.load(open(p))


def split_files(split_dir):
    return [f for s in SPLITS for f in sorted(glob.glob(os.path.join(split_dir, s, "*.parquet")))]


def split_caps(cfg):
    """Papers per split: all of them, or with max_papers the same share of every split (in its stored random
    order, so a smaller tier is a random, equally balanced subset of the stored split)."""
    sizes = {s: load_manifest(cfg.split_dir)["selected"][s]["total"] for s in SPLITS}
    total = sum(sizes.values())
    if not cfg.max_papers or cfg.max_papers >= total:
        return sizes
    return {s: max(1, round(cfg.max_papers * n / total)) if n else 0 for s, n in sizes.items()}


def iter_stored(cfg, columns):
    """Records of a stored split, each with its "_split"."""
    import pyarrow.parquet as pq
    caps = split_caps(cfg)
    for s in SPLITS:
        for f in sorted(glob.glob(os.path.join(cfg.split_dir, s, "*.parquet"))):
            pf = pq.ParquetFile(f)
            cols = [c for c in list(columns) + ["_split", "_split_rank"] if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=512, columns=cols):
                for rec in batch.to_pylist():
                    if rec["_split_rank"] < caps[s]:
                        yield rec


# ---------------------------------------------------------------- command line
def parse_args(argv=None):
    p = argparse.ArgumentParser("sci-jepa split", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", default=os.environ.get("PAPER_A_PARQUET", "data/normalized/arxiv"),
                   help="normalized parquet folder(s), comma-separated (e.g. .../normalized/arxiv,.../normalized/pubmed)")
    p.add_argument("--out", required=True, help="output folder, e.g. data/splits/d0")
    p.add_argument("--train-until", type=int, default=2023, help="train: papers up to this year (default 2023)")
    p.add_argument("--val-years", type=int, nargs="+", default=[2024], help="validation years (default 2024)")
    p.add_argument("--test-years", type=int, nargs="+", default=[2025, 2026], help="test years (default 2025 2026)")
    p.add_argument("--train-papers", type=int, default=60000, help="papers in train (0 = every usable paper, unbalanced)")
    p.add_argument("--val-papers", type=int, default=5000, help="papers in validation (0 = all)")
    p.add_argument("--test-papers", type=int, default=5000, help="papers in test (0 = all)")
    p.add_argument("--balance", default="temperature", choices=["temperature", "uniform", "natural", "none"],
                   help="domain shares, the same in every split: temperature (count^alpha), uniform (equal), natural "
                        "(pooled counts); none = each split keeps its own domain mix")
    p.add_argument("--alpha", type=float, default=0.5, help="temperature exponent (0.5 = square root, as for training)")
    p.add_argument("--ku-fixes", default=os.environ.get("PAPER_A_KU_FIXES"),
                   help="laion/Scientific-Summaries-KU-Fixes folder: drops duplicate copies and near-empty summaries")
    p.add_argument("--duplicates", default="all", choices=["all", "strong", "none"],
                   help="duplicate tier to drop (all = strong + probable, ~1%%; strong ~0.06%%)")
    p.add_argument("--subsets", nargs="*", default=None, help="only these subset values (e.g. arxiv-new titled)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rows-per-part", type=int, default=2000, help="papers per output parquet file")
    a = p.parse_args(argv)
    if os.path.exists(os.path.join(a.out, MANIFEST)):
        sys.exit(f"{a.out} already holds a split; choose another --out or delete it first")
    if not a.ku_fixes:
        print("[split] note: no --ku-fixes: duplicate copies are not removed (a paper may sit in two splits)")
    return a


def main(argv=None):
    make_split(parse_args(argv))


if __name__ == "__main__":
    main()
