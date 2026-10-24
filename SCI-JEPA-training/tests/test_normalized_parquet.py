"""
pytest suite for normalize_parquet.py

Part 1 (TestOutputFolder) validates the JSON files you actually produced:
       every file, every key, no missing, no extra, no leftover variants.
Part 2 (TestNormalizer / TestEndToEnd) tests the logic itself on synthetic
       parquet files built in a tmp dir -- these run with no data present.

Self-contained: needs NO conftest.py.

Run (from the repository root):
    python -m pytest -q tests/test_normalized_parquet.py
    python -m pytest -q tests/test_normalized_parquet.py -m logic     # logic only, no data

To validate a different folder, set an env var instead of a CLI flag:
    JSON_DIR=./some_other_folder python -m pytest -q tests/test_normalized_parquet.py
"""

import glob
import json
import os
import sys

import pytest

# scripts/data/normalize_parquet.py is a script, not part of the package: import it from its folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "data"))

from normalize_parquet import (          # noqa: E402
    CANONICAL_KEYS,
    _canon,
    _safe_filename,
    normalize_record,
)

# optional/bookkeeping keys the script adds on purpose
META_KEYS = {"_extra_fields", "_source_file"}
CANONICAL_SET = set(CANONICAL_KEYS)

# folder to validate -- override with:  OUT_DIR=./other python -m pytest ...
# (JSON_DIR is still honoured for compatibility)
JSON_DIR = os.environ.get("OUT_DIR") or os.environ.get("JSON_DIR", "./normalized_json")
PARQUET_DIR = JSON_DIR


@pytest.fixture(scope="session")
def json_files():
    files = sorted(glob.glob(os.path.join(JSON_DIR, "**", "*.json"), recursive=True))
    files += sorted(glob.glob(os.path.join(JSON_DIR, "**", "*.jsonl"), recursive=True))
    # bookkeeping files such as _progress.json are not records
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    if not files:
        pytest.skip(f"no json files in {JSON_DIR!r} — run normalize_parquet.py first")
    return files


def _records_of(path):
    """Yield every record in a .json (object or list) or .jsonl file."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    return data if isinstance(data, list) else [data]


# ============================================================ PART 1: my data
class TestOutputFolder:
    """Validates the real output folder."""

    def test_folder_has_files(self, json_files):
        assert len(json_files) > 0

    def test_every_file_is_valid_json(self, json_files):
        broken = []
        for path in json_files:
            try:
                _records_of(path)
            except Exception as exc:
                broken.append(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")
        assert not broken, "unparseable files:\n" + "\n".join(broken)

    def test_no_record_is_missing_a_canonical_key(self, json_files):
        """THE key check: every record must carry all 44 keys."""
        problems = []
        for path in json_files:
            for i, rec in enumerate(_records_of(path)):
                missing = CANONICAL_SET - set(rec)
                if missing:
                    problems.append(
                        f"{os.path.basename(path)}[{i}] missing {sorted(missing)}")
        assert not problems, "records missing keys:\n" + "\n".join(problems[:20])

    def test_no_record_has_unexpected_top_level_keys(self, json_files):
        """Nothing outside the 44 canonical keys + the two meta keys."""
        problems = []
        for path in json_files:
            for i, rec in enumerate(_records_of(path)):
                extra = set(rec) - CANONICAL_SET - META_KEYS
                if extra:
                    problems.append(
                        f"{os.path.basename(path)}[{i}] unexpected {sorted(extra)}")
        assert not problems, ("unexpected keys (should be inside _extra_fields):\n"
                              + "\n".join(problems[:20]))

    def test_all_records_share_identical_key_sets(self, json_files):
        """Every record must have the SAME keys as every other record."""
        signatures = {}
        for path in json_files:
            for i, rec in enumerate(_records_of(path)):
                sig = frozenset(set(rec) - META_KEYS)
                signatures.setdefault(sig, []).append(f"{os.path.basename(path)}[{i}]")

        if len(signatures) > 1:
            biggest = max(signatures, key=lambda s: len(signatures[s]))
            lines = []
            for sig, where in signatures.items():
                if sig == biggest:
                    continue
                lines.append(
                    f"{len(where)} record(s) differ, e.g. {where[0]}: "
                    f"missing={sorted(biggest - sig)} extra={sorted(sig - biggest)}")
            pytest.fail(f"{len(signatures)} distinct key sets found:\n" + "\n".join(lines))

    def test_key_order_is_canonical(self, json_files):
        """The 44 canonical keys should appear in CANONICAL_KEYS order."""
        problems = []
        for path in json_files:
            for i, rec in enumerate(_records_of(path)):
                order = [k for k in rec if k in CANONICAL_SET]
                if order != CANONICAL_KEYS:
                    problems.append(f"{os.path.basename(path)}[{i}]")
        assert not problems, ("key order differs from CANONICAL_KEYS in "
                              f"{len(problems)} record(s): {problems[:5]}")

    def test_no_unnormalized_key_variants_remain(self, json_files):
        """No 'paperId' / 'Source Title' style leftovers anywhere."""
        offenders = []
        for path in json_files:
            for i, rec in enumerate(_records_of(path)):
                for key in rec:
                    if key in CANONICAL_SET or key in META_KEYS:
                        continue
                    if _canon(key) in {_canon(k) for k in CANONICAL_KEYS}:
                        offenders.append(f"{os.path.basename(path)}[{i}]: {key!r}")
        assert not offenders, ("variant spellings that should have been renamed:\n"
                               + "\n".join(offenders[:20]))

    def test_paper_id_present_and_unique(self, json_files):
        """paper_id should identify a record; duplicates are worth knowing about."""
        seen, dupes, empty = {}, [], []
        for path in json_files:
            for i, rec in enumerate(_records_of(path)):
                pid = rec.get("paper_id")
                where = f"{os.path.basename(path)}[{i}]"
                if pid is None or str(pid).strip() == "":
                    empty.append(where)
                elif pid in seen:
                    dupes.append(f"{pid!r}: {seen[pid]} and {where}")
                else:
                    seen[pid] = where

        assert not empty, f"records with no paper_id: {empty[:10]}"
        assert not dupes, "duplicate paper_id values:\n" + "\n".join(dupes[:10])

    def test_report_fill_rate(self, json_files):
        """Not an assertion — prints how often each key is actually populated."""
        total, filled = 0, {k: 0 for k in CANONICAL_KEYS}
        for path in json_files:
            for rec in _records_of(path):
                total += 1
                for k in CANONICAL_KEYS:
                    if rec.get(k) not in (None, "", [], {}):
                        filled[k] += 1

        print(f"\nfill rate across {total} records:")
        for k in CANONICAL_KEYS:
            pct = 100 * filled[k] / total if total else 0
            flag = "  <-- always empty" if filled[k] == 0 else ""
            print(f"  {k:<32} {filled[k]:>6}/{total} ({pct:5.1f}%){flag}")
        assert total > 0


# ====================================================== PART 2: logic (no data)

@pytest.fixture(scope="session")
def parquet_files():
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, "**", "*.parquet"),
                             recursive=True))
    if not files:
        pytest.skip(f"no parquet output in {PARQUET_DIR!r} — "
                    f"run normalize_parquet.py with OUT_MODE='parquet' first")
    return files


class TestParquetOutput:
    """Validates OUT_MODE='parquet' output: the schema must be identical
    across every file, which is what makes a combined read back safe."""

    def test_every_file_opens(self, parquet_files):
        pq = pytest.importorskip("pyarrow.parquet")
        broken = []
        for path in parquet_files:
            try:
                pq.ParquetFile(path).metadata
            except Exception as exc:
                broken.append(f"{os.path.basename(path)}: {exc}")
        assert not broken, "unreadable parquet files:\n" + "\n".join(broken)

    def test_all_files_share_one_schema(self, parquet_files):
        """The input shards disagreed; the OUTPUT must not."""
        pq = pytest.importorskip("pyarrow.parquet")
        schemas = {}
        for path in parquet_files:
            sig = tuple((f.name, str(f.type)) for f in pq.ParquetFile(path).schema_arrow)
            schemas.setdefault(sig, []).append(os.path.basename(path))
        assert len(schemas) == 1, (
            "output files have differing schemas, so a combined read is unsafe: "
            + "; ".join(f"{len(v)} file(s) like {v[0]}" for v in schemas.values()))

    def test_schema_has_every_canonical_key(self, parquet_files):
        pq = pytest.importorskip("pyarrow.parquet")
        names = {f.name for f in pq.ParquetFile(parquet_files[0]).schema_arrow}
        missing = CANONICAL_SET - names
        assert not missing, f"columns missing from output schema: {sorted(missing)}"

    def test_combined_read_loses_no_rows(self, parquet_files):
        """pq.read_table(list) must now be safe -- no silent all-null rows."""
        pq = pytest.importorskip("pyarrow.parquet")
        per_file = sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)
        combined = pq.read_table(parquet_files)
        assert combined.num_rows == per_file, (
            f"combined read gave {combined.num_rows} rows, files hold {per_file}")

        ids = combined.column("paper_id").to_pylist()
        assert sum(1 for i in ids if i is None) == 0, (
            "null paper_ids in a combined read: the silent-schema bug is back")

    def test_paper_ids_unique_across_files(self, parquet_files):
        pq = pytest.importorskip("pyarrow.parquet")
        from collections import Counter
        ids = []
        for path in parquet_files:
            ids += pq.read_table(path, columns=["paper_id"]).column("paper_id").to_pylist()
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        assert not dupes, f"duplicate paper_ids: {dupes[:10]}"


def _record_files(folder):
    """*.json in folder, excluding bookkeeping files like _progress.json."""
    return sorted(p for p in folder.glob("*.json") if not p.name.startswith("_"))

logic = pytest.mark.logic      # registered via pytest.ini; harmless if absent


@logic
class TestNormalizer:
    def test_output_has_exactly_canonical_keys(self):
        out, _, _ = normalize_record({"paper_id": "p1"})
        assert [k for k in out if k != "_extra_fields"] == CANONICAL_KEYS

    @pytest.mark.parametrize("variant,expected", [
        ("paperId", "paper_id"),
        ("PAPER_ID", "paper_id"),
        ("Paper Id", "paper_id"),
        ("paper-id", "paper_id"),
        ("Source Title", "source_title"),
        ("FIELD-SUBFIELD", "field_subfield"),
        ("threeTakeaways", "three_takeaways"),
        ("oa_cited_by_count", "oa_cited_by_count"),
    ])
    def test_spelling_variants_are_renamed(self, variant, expected):
        out, unmatched, _ = normalize_record({variant: "V"})
        assert out[expected] == "V"
        assert not unmatched

    def test_missing_keys_become_none(self):
        out, _, _ = normalize_record({"paper_id": "p1"})
        assert out["text_raw"] is None
        assert sum(v is None for k, v in out.items() if k in CANONICAL_SET) == 43

    def test_unknown_key_is_preserved_not_dropped(self):
        out, unmatched, _ = normalize_record({"paper_id": "p", "mystery": 7})
        assert unmatched == ["mystery"]
        assert out["_extra_fields"] == {"mystery": 7}
        assert "mystery" not in set(out) - {"_extra_fields"}

    def test_near_miss_is_not_fuzzy_matched(self):
        """'tittle' must NOT be guessed into source_title."""
        out, unmatched, _ = normalize_record({"tittle": "X"})
        assert out["source_title"] is None
        assert unmatched == ["tittle"]

    def test_real_value_wins_over_empty_duplicate(self):
        out, _, _ = normalize_record({"paper_id": "", "paperId": "real"})
        assert out["paper_id"] == "real"

    def test_renames_are_reported(self):
        _, _, renames = normalize_record({"paperId": "p", "paper_id": None})
        assert ("paperId", "paper_id") in renames

    @pytest.mark.parametrize("raw,expected", [
        ("10.1234/abc.99", "10.1234_abc.99"),
        ("plain_id", "plain_id"),
        ("", "FB"),
        (None, "FB"),
        ("a/b\\c:d", "a_b_c_d"),
    ])
    def test_filename_sanitizing(self, raw, expected):
        assert _safe_filename(raw, "FB") == expected

    def test_filename_length_capped(self):
        assert len(_safe_filename("x" * 500, "FB")) <= 150


@logic
class TestEndToEnd:
    @pytest.fixture
    def mixed_parquet(self, tmp_path):
        """Two shards with DIFFERENT column names + a duplicate and a null id."""
        pd = pytest.importorskip("pandas")
        d = tmp_path / "pq"
        d.mkdir()
        pd.DataFrame({"paper_id": ["p1", None],
                      "Source Title": ["A", "B"]}).to_parquet(d / "s0.parquet")
        pd.DataFrame({"paperId": ["p1", "p3"],
                      "source_title": ["dup", "C"],
                      "junk_col": ["x", "y"]}).to_parquet(d / "s1.parquet")
        return d

    def test_mismatched_shards_lose_no_rows(self, mixed_parquet, monkeypatch):
        """Guards the pq.read_table(list) silent-null-row bug."""
        import normalize_parquet as mod
        monkeypatch.setattr(mod, "OUT_PATH", None)
        records = mod.read_parquet_folder(str(mixed_parquet), keep_in_memory=True)

        assert len(records) == 4, "rows were dropped or nulled out"
        titles = {r["source_title"] for r in records}
        assert titles == {"A", "B", "dup", "C"}

    def test_every_record_normalized_identically(self, mixed_parquet, monkeypatch):
        import normalize_parquet as mod
        monkeypatch.setattr(mod, "OUT_PATH", None)
        records = mod.read_parquet_folder(str(mixed_parquet), keep_in_memory=True)

        sigs = {frozenset(set(r) - META_KEYS) for r in records}
        assert len(sigs) == 1, "records came out with differing key sets"
        assert sigs.pop() == CANONICAL_SET

    def test_per_record_writes_one_file_per_row(self, mixed_parquet, tmp_path, monkeypatch):
        import normalize_parquet as mod
        out = tmp_path / "out"
        monkeypatch.setattr(mod, "OUT_PATH", str(out))
        monkeypatch.setattr(mod, "OUT_MODE", "per_record")
        records = mod.read_parquet_folder(str(mixed_parquet), keep_in_memory=True)

        written = [p.name for p in _record_files(out)]
        assert len(written) == len(records) == 4, written
        # duplicate id suffixed, null id fell back to shard+row
        assert "p1.json" in written and "p1__1.json" in written
        assert any(n.startswith("s0_row") for n in written), written

    def test_nothing_accumulates_in_memory_by_default(self, mixed_parquet, tmp_path,
                                                      monkeypatch):
        """Memory regression: the default run must NOT hold records in RAM.

        Accumulating every record is what OOM-killed the kernel originally.
        """
        import normalize_parquet as mod
        out = tmp_path / "out"
        monkeypatch.setattr(mod, "OUT_PATH", str(out))
        monkeypatch.setattr(mod, "OUT_MODE", "per_record")
        monkeypatch.setattr(mod, "KEEP_IN_MEMORY", False)

        returned = mod.read_parquet_folder(str(mixed_parquet))

        assert returned == [], "records are being accumulated in memory again"
        # but every row still reached the disk
        assert len(_record_files(out)) == 4

    def test_drop_columns_still_emits_full_schema(self, mixed_parquet, tmp_path,
                                                  monkeypatch):
        """A dropped column must come out as None, not vanish from the keys."""
        import normalize_parquet as mod
        out = tmp_path / "out"
        monkeypatch.setattr(mod, "OUT_PATH", str(out))
        monkeypatch.setattr(mod, "OUT_MODE", "per_record")
        monkeypatch.setattr(mod, "DROP_COLUMNS", ["source_title"])

        mod.read_parquet_folder(str(mixed_parquet))

        for path in _record_files(out):
            rec = json.loads(path.read_text(encoding="utf-8"))
            assert CANONICAL_SET <= set(rec)
            assert rec["source_title"] is None

    def test_written_files_round_trip(self, mixed_parquet, tmp_path, monkeypatch):
        import normalize_parquet as mod
        out = tmp_path / "out"
        monkeypatch.setattr(mod, "OUT_PATH", str(out))
        monkeypatch.setattr(mod, "OUT_MODE", "per_record")
        mod.read_parquet_folder(str(mixed_parquet))

        for path in _record_files(out):
            rec = json.loads(path.read_text(encoding="utf-8"))
            assert CANONICAL_SET <= set(rec)
            assert [k for k in rec if k in CANONICAL_SET] == CANONICAL_KEYS