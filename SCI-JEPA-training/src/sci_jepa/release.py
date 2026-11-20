"""
Release package of a trained model (sci-jepa release): the controls the regulation assessment asks for, applied when
anything leaves the project storage.

  export   weights + config of one arm's best seed (best validation), MODEL_CARD.md, DATASHEET.md, NOTICE (dataset
           attribution, component licences), LICENSE (the chosen licence, see docs/governance/LICENCE_DECISION.md),
           optionally the KU texts of the released split: personal data redacted (e-mail, phone, ORCID), papers on the
           exclusion list (retracted, withdrawn, removal requests) dropped; never the papers' full texts
  check    a release folder: no full texts, no e-mail / phone / ORCID, NOTICE and LICENSE present -> pass / problems

    sci-jepa release export <results folder> <dest> [--arm jepa+rgat] [--with-texts]
    sci-jepa release check <dest>
"""
import argparse
import glob
import json
import os
import re
import shutil

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<![\w.])\+\d{1,3}[ (.-]{0,2}\d{2,4}[ ).-]{1,2}\d{3,4}[ .-]\d{3,4}(?![\w.])")
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
FULLTEXT_KEYS = ("text_raw", "text_sanitized")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def redact(text):
    """Personal contact data replaced by placeholders; everything else unchanged."""
    if not text:
        return text
    return ORCID_RE.sub("<orcid>", PHONE_RE.sub("<phone>", EMAIL_RE.sub("<email>", text)))


def exclusions(path=None):
    path = path or os.path.join(PROJ, "release", "exclude_papers.txt")
    if not os.path.exists(path):
        return set()
    return {ln.split("#")[0].strip() for ln in open(path) if ln.split("#")[0].strip()}


def _best_seed(exp_dir, arm):
    best, score = None, None
    for f in sorted(glob.glob(os.path.join(exp_dir, f"{arm}_seed*.json"))):
        m = json.load(open(f))
        s = (m.get("history") or {}).get("best_val") if isinstance(m.get("history"), dict) else None
        R = m.get("epoch_results") or []
        s = s if s is not None else max([r.get("val_ratio") or r.get("ratio") or 0 for r in R] or [0])
        if m.get("model_dir") and (score is None or s > score):
            best, score = m, s
    return best


def export(out_dir, dest, arm="jepa+rgat", exp="BENCH", with_texts=False, log=print):
    exp_dir = os.path.join(out_dir, exp)
    m = _best_seed(exp_dir, arm)
    if m is None:
        raise FileNotFoundError(f"no trained {arm} model in {exp_dir}")
    os.makedirs(dest, exist_ok=True)
    md = m["model_dir"]
    for f in ("best.safetensors", "config.json"):
        if os.path.exists(os.path.join(md, f)):
            shutil.copy(os.path.join(md, f), os.path.join(dest, "model.safetensors" if f == "best.safetensors" else f))
    for src, name in ((os.path.join(PROJ, "docs", "MODEL_CARD.md"), "README.md"),
                      (os.path.join(PROJ, "docs", "DATASHEET.md"), "DATASHEET.md"),
                      (os.path.join(PROJ, "release", "NOTICE"), "NOTICE"),
                      (os.path.join(PROJ, "release", "LICENSE"), "LICENSE")):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest, name))
    if with_texts:
        import pyarrow.parquet as pq
        from sci_jepa.data.ku_graph import READ_COLS
        ex = exclusions()
        split = (m.get("config") or {}).get("split_dir")
        n = dropped = 0
        with open(os.path.join(dest, "ku_texts.jsonl"), "w") as fo:
            for f in sorted(glob.glob(os.path.join(split, "*", "*.parquet"))):
                pf = pq.ParquetFile(f)
                cols = [c for c in READ_COLS if c in pf.schema_arrow.names and c not in FULLTEXT_KEYS]
                for b in pf.iter_batches(columns=cols, batch_size=512):
                    for r in b.to_pylist():
                        if str(r.get("paper_id")) in ex:
                            dropped += 1; continue
                        fo.write(json.dumps({k: redact(v) if isinstance(v, str) else v for k, v in r.items()}) + "\n")
                        n += 1
        log(f"[release] KU texts: {n} papers written, {dropped} excluded, personal data redacted")
    rep = check(dest)
    log(f"[release] {arm} ({os.path.basename(md)}) -> {dest}: {'pass' if rep['ok'] else rep['problems']}")
    return rep


def check(dest):
    """No full texts, no contact data, NOTICE and LICENSE present."""
    problems = []
    for need in ("NOTICE", "LICENSE", "README.md"):
        if not os.path.exists(os.path.join(dest, need)):
            problems.append(f"missing {need}")
    for f in glob.glob(os.path.join(dest, "**", "*"), recursive=True):
        if not os.path.isfile(f) or f.endswith(".safetensors"):
            continue
        with open(f, errors="ignore") as fh:
            for i, line in enumerate(fh):
                if any(f'"{k}"' in line for k in FULLTEXT_KEYS):
                    problems.append(f"{os.path.basename(f)}:{i + 1} contains a full-text field"); break
                if EMAIL_RE.search(line) or ORCID_RE.search(line) or PHONE_RE.search(line):
                    if os.path.basename(f) not in ("NOTICE", "LICENSE", "README.md"):   # contact of the maintainers
                        problems.append(f"{os.path.basename(f)}:{i + 1} contains contact data"); break
    return {"ok": not problems, "problems": problems[:20]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("out"); e.add_argument("dest")
    e.add_argument("--arm", default="jepa+rgat"); e.add_argument("--exp", default="BENCH")
    e.add_argument("--with-texts", action="store_true")
    c = sub.add_parser("check"); c.add_argument("dest")
    A = p.parse_args(argv)
    if A.cmd == "export":
        export(A.out, A.dest, A.arm, A.exp, A.with_texts)
    else:
        r = check(A.dest)
        print("pass" if r["ok"] else "\n".join(r["problems"]))


if __name__ == "__main__":
    main()
