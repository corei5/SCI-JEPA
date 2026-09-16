"""
Download the raw paper corpus from HuggingFace into dataset_creation/data/raw/.

Standard library only — no `datasets`, no `huggingface_hub`, nothing to pip
install. It talks to the HF REST API directly with urllib.

Token resolution, first hit wins:
    1. --token
    2. $HF_TOKEN
    3. $HUGGING_FACE_HUB_TOKEN
    4. ~/.cache/huggingface/token

`ai4sci-tib/LAION_arxiv-open` is a GATED dataset, so a token is required and the
account behind it must have been granted access on the dataset page.

Usage
-----
    python -m dataset_creation.fetch_hf --list           # what's in the repo
    python -m dataset_creation.fetch_hf                  # download everything
    python -m dataset_creation.fetch_hf --limit 500      # just the first 500 files
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "data", "raw")

REPO_ID = "ai4sci-tib/LAION_arxiv-open"
API = "https://huggingface.co/api/datasets"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


# ---------------------------------------------------------------------------
def resolve_token(explicit=None):
    """Find a usable HF token, or return None."""
    if explicit:
        return explicit.strip()
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            val = fh.read().strip()
        if val:
            return val
    return None


def _request(url, token, binary=False):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "sci-jepa-dataset-creation/1.0")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    return data if binary else json.loads(data.decode("utf-8"))


def _explain(err, token):
    """Turn an HTTP error into something actionable."""
    if not isinstance(err, urllib.error.HTTPError):
        return f"network error: {err}"
    if err.code == 401:
        if token:
            return ("401 Unauthorized — the token was rejected. If it is long "
                    "(hundreds of chars) it is a stale OAuth token, not a user "
                    "access token. Create one at "
                    "https://huggingface.co/settings/tokens (it looks like "
                    "'hf_...') and export it as HF_TOKEN.")
        return ("401 Unauthorized — no token found. Export HF_TOKEN, or pass "
                "--token.")
    if err.code == 403:
        return (f"403 Forbidden — the token is valid but this account has not "
                f"been granted access to {REPO_ID}. Request access on its "
                f"dataset page, then retry.")
    if err.code == 404:
        return (f"404 — {REPO_ID} not found at that revision. Check the repo id "
                f"and --revision.")
    return f"HTTP {err.code}: {err.reason}"


# ---------------------------------------------------------------------------
def list_files(repo, revision, token, suffix=".json"):
    """Return every file in the dataset repo matching `suffix`."""
    url = f"{API}/{repo}/tree/{urllib.parse.quote(revision)}?recursive=1"
    entries, cursor = [], url
    while cursor:
        try:
            batch = _request(cursor, token)
        except Exception as ex:
            raise SystemExit(f"[fetch_hf] cannot list {repo}: {_explain(ex, token)}")
        if isinstance(batch, dict):
            batch = batch.get("tree", [])
        entries.extend(batch)
        cursor = None                       # tree?recursive=1 returns the whole tree
    files = [e["path"] for e in entries
             if e.get("type") == "file" and e["path"].endswith(suffix)]
    return sorted(files)


def download(repo, revision, paths, out_dir, token, skip_existing=True):
    os.makedirs(out_dir, exist_ok=True)
    done, failed = 0, []
    total = len(paths)
    for i, path in enumerate(paths, 1):
        dest = os.path.join(out_dir, os.path.basename(path))
        if skip_existing and os.path.exists(dest) and os.path.getsize(dest) > 0:
            done += 1
            continue
        url = RESOLVE.format(repo=repo, rev=urllib.parse.quote(revision),
                             path=urllib.parse.quote(path))
        try:
            blob = _request(url, token, binary=True)
        except Exception as ex:
            failed.append((path, str(ex)))
            continue
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:            # write-then-rename: never leave a
            fh.write(blob)                     # half-written file that looks valid
        os.replace(tmp, dest)
        done += 1
        if i % 50 == 0 or i == total:
            print(f"[fetch_hf] {i}/{total} files", flush=True)
    return done, failed


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--token", default=None, help="overrides HF_TOKEN / token file")
    ap.add_argument("--limit", type=int, default=None, help="download only the first N files")
    ap.add_argument("--suffix", default=".json")
    ap.add_argument("--list", action="store_true", help="list files and exit")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    token = resolve_token(args.token)
    if not token:
        sys.exit("[fetch_hf] no token found. Create a user access token at "
                 "https://huggingface.co/settings/tokens and `export HF_TOKEN=hf_...`")
    if len(token) > 200:
        print("[fetch_hf] WARNING: that token is unusually long — it looks like a "
              "stale OAuth token rather than an 'hf_...' user access token. "
              "Expect a 401.", file=sys.stderr)

    files = list_files(args.repo, args.revision, token, args.suffix)
    print(f"[fetch_hf] {args.repo}@{args.revision}: {len(files)} '{args.suffix}' files")
    if args.list:
        for f in files[:50]:
            print("   ", f)
        if len(files) > 50:
            print(f"    … and {len(files) - 50} more")
        return

    if args.limit:
        files = files[:args.limit]
    done, failed = download(args.repo, args.revision, files, args.out, token,
                            skip_existing=not args.force)
    print(f"[fetch_hf] downloaded/present: {done}/{len(files)} -> {args.out}")
    if failed:
        print(f"[fetch_hf] {len(failed)} failed; first few:")
        for path, ex in failed[:5]:
            print(f"    {path}: {ex}")
    print(f"\nNext:\n  python -m dataset_creation.build_sample "
          f"--raw-dir {args.out} --out dataset_creation/data/kg --snippet 400")


if __name__ == "__main__":
    main()
