"""
Read every .parquet file in a folder, normalize each row onto one fixed
schema, and store the result as JSON -- in CONSTANT memory.

Why each shard is read SEPARATELY:
    pq.read_table([f1, f2]) applies the FIRST file's schema to all of them.
    If f2 names its columns differently, its rows come back as all-None
    WITHOUT raising an error -- silent data loss. Reading file by file and
    normalizing each one is the only safe way when schemas vary.

Why this no longer OOM-kills the kernel:
    1. Rows stream in small batches via iter_batches(), so a whole shard is
       never converted to Python objects at once (to_pylist() on a full table
       was costing several times the file size in RAM).
    2. Records are written to disk and DROPPED immediately. Nothing is
       accumulated across files -- KEEP_IN_MEMORY is False by default. The old
       `all_records.extend(...)` held the entire dataset in RAM; that was the
       main cause of the crash.
    3. After each shard, the Arrow table is deleted, gc.collect() runs, and
       Arrow's internal buffer pool is released back to the OS, so memory
       returns to baseline between files.
    4. "jsonl"/"json" modes stream to disk too, instead of building a list.

Remaining limit: peak memory is set by the largest ROW GROUP, because parquet
cannot decompress less than one row group. Lowering BATCH_SIZE below ~64 will
not help with that; DROP_COLUMNS is the lever that does.
"""

import gc
import glob
import json
import os
import resource
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

# Parallel column readers each buffer their own decompressed chunk, which
# multiplies peak memory on wide rows. One IO thread keeps that flat.
pa.set_io_thread_count(1)

# ----------------------------------------------------------------------------
# settings
#
# Edit the values here, OR override any of them with an environment variable of
# the same name:
#     PARQUET_FOLDER=data/raw/arxiv OUT_PATH=data/normalized/arxiv python scripts/data/normalize_parquet.py
# ----------------------------------------------------------------------------
def _env(name, default):
    """Environment variable wins over the default written here, if set."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    if isinstance(default, list):
        return [p.strip() for p in raw.split(",") if p.strip()]
    return raw


PARQUET_FOLDER = _env("PARQUET_FOLDER", "./data/raw/data/arxiv")

OUT_PATH = _env("OUT_PATH", "./data/normalized/arxiv")
# OUTPUT FOLDER (created if missing); set OUT_PATH="" in the env for a dry run
OUT_PATH = OUT_PATH or None

OUT_MODE = _env("OUT_MODE", "parquet")
                                  # "parquet"    -> one normalized .parquet per shard
                                  #                 (smallest on disk, easiest to load)
                                  # "per_record" -> one .json PER ROW, named by paper_id
                                  # "per_file"   -> one .json per parquet shard
                                  # "jsonl"      -> OUT_PATH/all_records.jsonl (streamed)
                                  # "json"       -> OUT_PATH/all_records.json  (streamed)

PARQUET_COMPRESSION = _env("PARQUET_COMPRESSION", "zstd")      # zstd (smallest) | snappy (fastest) | gzip | none
PARQUET_ROW_GROUP = _env("PARQUET_ROW_GROUP", 256)           # rows per row group in the OUTPUT files. Small
                                  # groups keep reading them back memory-cheap.

NAME_FIELD = _env("NAME_FIELD", "paper_id")           # which canonical key names the per_record files

KEEP_EXTRAS = _env("KEEP_EXTRAS", True)                # unmatched columns -> "_extra_fields"
LOG_RENAMES = _env("LOG_RENAMES", True)                # print every  original -> canonical  rename

# ---- memory knobs ----------------------------------------------------------
BATCH_SIZE = _env("BATCH_SIZE", 64)                   # rows pulled into Python at once. Lower it if
                                  # memory is still tight (32, or even 8).

KEEP_IN_MEMORY = _env("KEEP_IN_MEMORY", False)            # False = records written then dropped (constant
                                  # memory). True is what caused the original OOM;
                                  # only use it on small datasets or in tests.

DROP_COLUMNS = _env("DROP_COLUMNS", [])                 # BIGGEST memory win available: columns listed
                                  # here are never read at all, e.g.
                                  #     DROP_COLUMNS = ["text_raw"]
                                  # They are still written out as None, so the
                                  # 44-key schema is unchanged. Matched the same
                                  # canonical way as the keys, so "text_raw" also
                                  # drops a shard spelling it "Text Raw".

COMPACT_JSON = _env("COMPACT_JSON", False)              # True = no indentation: smaller files, less
                                  # string building per record.

SHOW_MEMORY = _env("SHOW_MEMORY", True)                # print memory after each shard
MAX_FILES = _env("MAX_FILES", 0)                     # >0 = only process the first N shards (trial run)

SKIP_DONE = _env("SKIP_DONE", True)                  # resume: skip shards already recorded as finished
                                  # in OUT_PATH/_progress.json. If the kernel dies
                                  # again, just re-run -- completed shards are not
                                  # redone. Set False to force a full rebuild.
                                  # (Not supported for OUT_MODE="json".)

FSYNC_PER_SHARD = _env("FSYNC_PER_SHARD", True)            # force each shard's output to physical disk before
                                  # moving on, so a crash cannot lose finished work

CANONICAL_KEYS = [
    'paper_id', 'subset', 'source_title', 'source_authors', 'source_year',
    'source_venue', 'source_doi', 'summary_title', 'summary_authors',
    'field_subfield', 'type_of_paper', 'executive_summary', 'research_context',
    'research_question_hypothesis', 'methodological_details',
    'procedures_architectures', 'key_results', 'interpretation_implications',
    'contradictions_limitations', 'claims', 'data_code_availability',
    'robustness_ablation_notes', 'ethical_considerations', 'key_figures_tables',
    'three_takeaways', 'summary_model', 'summary_tokens_in',
    'summary_tokens_out', 'openalex_id', 'oa_doi', 'oa_title', 'oa_type',
    'oa_year', 'oa_cited_by_count', 'oa_language', 'oa_is_retracted',
    'oa_is_open_access', 'oa_authorships', 'oa_referenced_works',
    'match_method', 'has_openalex_match', 'text_length', 'text_sanitized',
    'text_raw',
]

# Only for columns whose WORDS differ from the canonical name (casing, spaces,
# hyphens and underscores are already handled automatically below).
# Empty by default on purpose: a wrong entry here silently misfiles data.
# Fill it in after reading the "matched nothing canonical" report.
MANUAL_ALIASES = {
    # "abstract": "executive_summary",
    # "full_text": "text_raw",
}


def _canon(name: str) -> str:
    """paperId / 'Source Title' / FIELD-SUBFIELD -> paperid / sourcetitle / fieldsubfield"""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


_LOOKUP = {_canon(k): k for k in CANONICAL_KEYS}
_LOOKUP.update({_canon(src): dst for src, dst in MANUAL_ALIASES.items()})


def normalize_record(record: dict):
    """-> (record with exactly CANONICAL_KEYS, unmatched keys, renames done)"""
    out = {k: None for k in CANONICAL_KEYS}
    extras, unmatched, renames = {}, [], []

    for key, value in record.items():
        target = _LOOKUP.get(_canon(key))
        if target is None:
            unmatched.append(key)
            extras[key] = value
            continue
        if key != target:
            renames.append((key, target))
        # never let an empty duplicate variant clobber a real value
        if out[target] is None or out[target] == "":
            out[target] = value

    if KEEP_EXTRAS and extras:
        out["_extra_fields"] = extras
    return out, unmatched, renames


def _safe_filename(value, fallback: str) -> str:
    """paper_id -> a filesystem-safe stem; falls back when missing/unusable."""
    if value is None or str(value).strip() == "":
        return fallback
    stem = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(value))
    stem = stem.strip("._") or fallback
    return stem[:150]          # keep well under filesystem name limits


# ----------------------------------------------------------------------------
# memory helpers
# ----------------------------------------------------------------------------
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    _PROC = None               # optional; falls back to peak-only reporting


def rss_mb() -> float:
    """CURRENT resident memory in MB (falls back to peak if psutil absent)."""
    if _PROC is not None:
        return _PROC.memory_info().rss / (1024 * 1024)
    return peak_mb()


def peak_mb() -> float:
    """High-water mark RSS in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def free_arrow_memory():
    """Collect Python garbage, then hand Arrow's cached buffers back to the OS."""
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def build_unified_schema(files):
    """One output schema for ALL shards, derived from their footers only.

    Reading footers is cheap (no row data). Every output file then shares the
    same schema, so you CAN safely read them back together with
    pq.read_table(folder) -- which is exactly what is unsafe about the input
    shards, whose column names disagree.

    Type resolution per canonical key:
        all shards agree       -> that type
        shards disagree        -> string (values are stringified)
        key absent everywhere  -> null  (column exists, all values None)
    """
    types = {}                         # canonical key -> set of arrow types
    for path in files:
        try:
            schema = pq.ParquetFile(path).schema_arrow
        except Exception:
            continue
        for field in schema:
            target = _LOOKUP.get(_canon(field.name))
            if target:
                types.setdefault(target, set()).add(field.type)

    fields, conflicts, empties = [], [], []
    for key in CANONICAL_KEYS:
        found = types.get(key, set())
        if not found:
            fields.append(pa.field(key, pa.null()))
            empties.append(key)
        elif len(found) == 1:
            fields.append(pa.field(key, next(iter(found))))
        else:
            fields.append(pa.field(key, pa.string()))
            conflicts.append((key, sorted(str(t) for t in found)))

    if KEEP_EXTRAS:
        fields.append(pa.field("_extra_fields", pa.string()))   # JSON-encoded
    fields.append(pa.field("_source_file", pa.string()))

    if conflicts:
        print("\ncolumns whose type differs between shards -> stored as string:")
        for key, tys in conflicts:
            print(f"  {key}: {tys}")
    if empties:
        print(f"\n{len(empties)} key(s) absent from every shard, stored as "
              f"all-null columns: {', '.join(empties[:8])}"
              f"{' ...' if len(empties) > 8 else ''}")

    return pa.schema(fields)


def _progress_path():
    return os.path.join(OUT_PATH, "_progress.json") if OUT_PATH else None


def load_progress():
    """-> (set of finished shard names, byte offset of the last good write).

    The offset matters for jsonl: a crash mid-shard leaves a partial run of
    lines in the file. Resuming must rewind to the end of the last COMPLETED
    shard, or those rows get written twice.
    """
    path = _progress_path()
    if not path or not os.path.exists(path):
        return set(), 0
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("done", [])), int(data.get("offset", 0))
    except Exception:
        return set(), 0            # unreadable manifest: just start over


def save_progress(done, rows_by_shard, offset=0):
    """Record finished shards atomically, so a crash leaves a valid manifest."""
    path = _progress_path()
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done), "rows": rows_by_shard,
                   "offset": offset}, f, indent=2)
        f.flush()
        if FSYNC_PER_SHARD:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    os.replace(tmp, path)          # atomic rename


def _dump_kwargs():
    kw = {"ensure_ascii": False, "default": str}
    if not COMPACT_JSON:
        kw["indent"] = 2
    return kw


# ----------------------------------------------------------------------------
# streaming writer -- writes each record and forgets it
# ----------------------------------------------------------------------------
class _Writer:
    def __init__(self, out_path, mode, resume=False, resume_offset=0, schema=None):
        self.out_path, self.mode = out_path, mode
        self.handle = None
        self.dest = None
        self.count = 0
        self.first = True
        self.used_names = set()
        self.schema = schema           # parquet mode only
        self.pq_writer = None
        self.pq_dest = None
        self.buffer = []               # rows waiting to become a row group

        if out_path and mode in ("jsonl", "json"):
            os.makedirs(out_path, exist_ok=True)
            ext = "jsonl" if mode == "jsonl" else "json"
            self.dest = os.path.join(out_path, f"all_records.{ext}")
            # Resuming jsonl: rewind to the end of the last COMPLETED shard and
            # continue from there, discarding any partial rows a crash left
            # behind. json (a single array) cannot be appended to, so it restarts.
            if resume and mode == "jsonl" and os.path.exists(self.dest):
                self.handle = open(self.dest, "r+", encoding="utf-8")
                self.handle.seek(resume_offset)
                self.handle.truncate()       # drop the partial shard's rows
                dropped = os.path.getsize(self.dest)
                print(f"  rewound {self.dest} to byte {resume_offset} "
                      f"(now {dropped} bytes) to drop partial rows")
            else:
                self.handle = open(self.dest, "w", encoding="utf-8")
                if mode == "json":
                    self.handle.write("[\n")
        elif out_path:
            os.makedirs(out_path, exist_ok=True)

    def offset(self):
        """Current byte position, recorded per shard so a resume can rewind."""
        return self.handle.tell() if self.handle else 0

    # --- parquet mode ------------------------------------------------------
    def open_parquet(self, shard_name):
        """One output .parquet per input shard."""
        if not self.out_path or self.mode != "parquet":
            return
        os.makedirs(self.out_path, exist_ok=True)
        self.pq_dest = os.path.join(self.out_path, shard_name)
        self.pq_writer = pq.ParquetWriter(
            self.pq_dest, self.schema,
            compression=None if PARQUET_COMPRESSION == "none" else PARQUET_COMPRESSION)

    def _flush_row_group(self):
        """Turn the buffered rows into one row group and release them."""
        if not self.buffer or self.pq_writer is None:
            return
        try:
            table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
            # a value did not fit its column type (mixed types across shards):
            # stringify the offending columns rather than lose the rows
            safe = []
            for row in self.buffer:
                out = {}
                for field in self.schema:
                    v = row.get(field.name)
                    if field.type == pa.string() and v is not None and not isinstance(v, str):
                        v = json.dumps(v, ensure_ascii=False, default=str)
                    out[field.name] = v
                safe.append(out)
            table = pa.Table.from_pylist(safe, schema=self.schema)
            del safe
        self.pq_writer.write_table(table)
        del table
        self.buffer.clear()

    def close_parquet(self):
        if self.pq_writer is None:
            return
        self._flush_row_group()
        self.pq_writer.close()
        self.pq_writer = None
        print(f"     -> {self.pq_dest}")

    # --- per-row modes -----------------------------------------------------
    def write_record(self, record, shard_name, row_index):
        if not self.out_path:
            return
        if self.mode == "parquet":
            row = dict(record)
            extras = row.pop("_extra_fields", None)
            if KEEP_EXTRAS:
                row["_extra_fields"] = (
                    json.dumps(extras, ensure_ascii=False, default=str)
                    if extras else None)
            self.buffer.append(row)
            if len(self.buffer) >= PARQUET_ROW_GROUP:
                self._flush_row_group()
            self.count += 1
            return
        if self.mode == "jsonl":
            self.handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        elif self.mode == "json":
            if not self.first:
                self.handle.write(",\n")
            json.dump(record, self.handle, **_dump_kwargs())
            self.first = False
        elif self.mode == "per_record":
            stem = _safe_filename(
                record.get(NAME_FIELD),
                fallback=f"{shard_name.replace('.parquet', '')}_row{row_index}")
            unique, n = stem, 1
            while unique in self.used_names:     # never overwrite a duplicate id
                unique = f"{stem}__{n}"
                n += 1
            self.used_names.add(unique)
            with open(os.path.join(self.out_path, f"{unique}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(record, f, **_dump_kwargs())
        else:
            return                               # per_file is written per shard
        self.count += 1

    def write_shard(self, records, shard_name):
        if not self.out_path or self.mode != "per_file" or not records:
            return
        dest = os.path.join(self.out_path, shard_name.replace(".parquet", ".json"))
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(records, f, **_dump_kwargs())
        self.count += len(records)
        print(f"     -> {dest}")

    def flush_shard(self):
        """Push this shard's output to disk so a later crash cannot lose it."""
        if self.handle:
            self.handle.flush()
            if FSYNC_PER_SHARD:
                try:
                    os.fsync(self.handle.fileno())
                except OSError:
                    pass

    def close(self):
        if self.mode == "parquet":
            self.close_parquet()
            if self.out_path and self.count:
                print(f"\nwrote {self.count} records -> {self.out_path}/*.parquet")
            return
        if self.handle:
            if self.mode == "json":
                self.handle.write("\n]\n")
            self.handle.close()
            print(f"\nwrote {self.count} records -> {self.dest}")
        elif self.out_path and self.count:
            print(f"\nwrote {self.count} json files -> {self.out_path}/")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def _check_paths(folder, out_path):
    """Refuse to write output into the input folder.

    In parquet mode each output file takes the INPUT shard's filename, so if
    out_path is the input folder (or contains it) the run would overwrite the
    source dataset in place. That is unrecoverable, so it is a hard error.
    """
    if not out_path:
        return
    src = os.path.realpath(folder)
    dst = os.path.realpath(out_path)
    if src == dst:
        raise SystemExit(
            f"\n*** REFUSING TO RUN: OUT_PATH is the same folder as "
            f"PARQUET_FOLDER:\n      {src}\n"
            f"    Output files reuse the input shard names, so this would "
            f"overwrite your source\n    data. Point OUT_PATH at a NEW, "
            f"separate folder.\n")
    if dst.startswith(src + os.sep) or src.startswith(dst + os.sep):
        print(f"\n*** WARNING: OUT_PATH and PARQUET_FOLDER are nested:\n"
              f"      input : {src}\n      output: {dst}\n"
              f"    A later run would pick up your own output as input. "
              f"Use a separate folder.\n")

    # existing parquet files in the output folder that we are NOT about to
    # overwrite are a sign OUT_PATH is pointing somewhere it should not
    if OUT_MODE == "parquet" and os.path.isdir(dst):
        existing = [f for f in os.listdir(dst) if f.endswith(".parquet")]
        inputs = {os.path.basename(p) for p in
                  glob.glob(f"{folder}/**/*.parquet", recursive=True)}
        foreign = [f for f in existing if f not in inputs]
        if foreign:
            print(f"*** WARNING: {len(foreign)} .parquet file(s) already in "
                  f"OUT_PATH that this run will NOT replace,\n"
                  f"    e.g. {foreign[:3]}. Is OUT_PATH the right folder? "
                  f"Reading it back later would mix them in.\n")


def read_parquet_folder(folder: str = None, keep_in_memory: bool = None):
    """Stream every shard, normalize, write. Returns [] unless keep_in_memory."""
    folder = folder or PARQUET_FOLDER
    keep = KEEP_IN_MEMORY if keep_in_memory is None else keep_in_memory

    _check_paths(folder, OUT_PATH)

    files = sorted(glob.glob(f"{folder}/**/*.parquet", recursive=True))
    if not files:
        print(f"*** No .parquet files found under {folder!r} — nothing to do.\n"
              f"    Check PARQUET_FOLDER. (Looked for "
              f"{folder}/**/*.parquet)")
        return []
    if MAX_FILES:
        files = files[:MAX_FILES]
    print(f"found {len(files)} parquet files   (batch={BATCH_SIZE}, "
          f"mode={OUT_MODE}, keep_in_memory={keep})")

    # ---- resume support: which shards did a previous run already finish? ----
    resume = SKIP_DONE and OUT_MODE != "json"
    done, resume_offset = load_progress() if resume else (set(), 0)
    rows_by_shard = {}
    if done:
        print(f"resuming: {len(done)} shard(s) already done, skipping them "
              f"(set SKIP_DONE = False to rebuild everything)")

    # one schema shared by every output file, from footers only (cheap)
    schema = build_unified_schema(files) if OUT_MODE == "parquet" else None

    writer = _Writer(OUT_PATH, OUT_MODE, resume=bool(done),
                     resume_offset=resume_offset, schema=schema)
    kept = []
    unmatched_counter, rename_counter, missing_counter = Counter(), Counter(), Counter()
    total_rows = 0
    first_example = None

    for fi, path in enumerate(files, 1):
        name = os.path.basename(path)

        if name in done:
            print(f"  [{fi}/{len(files)}] {name}: skipped (already done)")
            continue

        try:
            pf = pq.ParquetFile(path)            # reads the footer only
        except Exception as exc:
            print(f"  !! skipped {name}: {type(exc).__name__}: {exc}")
            continue

        n_rows = pf.metadata.num_rows
        available = list(pf.schema_arrow.names)
        n_cols = len(available)

        # match DROP_COLUMNS canonically, like the key renaming does
        read_cols = None
        if DROP_COLUMNS:
            drop = {_canon(c) for c in DROP_COLUMNS}
            read_cols = [c for c in available if _canon(c) not in drop]

        # peak memory is set by the row group, not BATCH_SIZE: say so once
        if fi == 1 and pf.metadata.num_row_groups:
            rg_rows = pf.metadata.row_group(0).num_rows
            if rg_rows > 50 * BATCH_SIZE:
                print(f"  note: row groups hold {rg_rows} rows, so peak memory is "
                      f"set by the row group, not BATCH_SIZE={BATCH_SIZE}.\n"
                      f"        If memory is still tight, use DROP_COLUMNS to skip "
                      f"large fields you don't need.")

        shard_records = [] if OUT_MODE == "per_file" else None
        row_index = 0

        if OUT_MODE == "parquet":
            writer.open_parquet(name)      # one output .parquet for this shard

        try:
            # stream BATCH_SIZE rows at a time instead of the whole shard
            for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                         columns=read_cols,
                                         use_threads=False):
                for raw in batch.to_pylist():
                    fixed, unmatched, renames = normalize_record(raw)
                    fixed["_source_file"] = name

                    unmatched_counter.update(unmatched)
                    rename_counter.update(renames)
                    missing_counter.update(
                        k for k in CANONICAL_KEYS if fixed[k] is None)

                    if first_example is None:    # keep a truncated copy to print
                        first_example = {
                            k: (v if not isinstance(v, str) or len(v) < 120
                                else v[:120] + f"... [{len(v)} chars]")
                            for k, v in fixed.items()}

                    writer.write_record(fixed, name, row_index)
                    if shard_records is not None:
                        shard_records.append(fixed)
                    if keep:
                        kept.append(fixed)

                    row_index += 1
                    total_rows += 1
                    del fixed                    # drop the record now
                del batch                        # drop the Arrow batch
        except Exception as exc:
            print(f"  !! error mid-file {name} at row {row_index}: "
                  f"{type(exc).__name__}: {exc}")

        # ---- 1. SAVE this shard's output ----
        if shard_records is not None:
            writer.write_shard(shard_records, name)
            shard_records.clear()
        if OUT_MODE == "parquet":
            writer.close_parquet()               # finalize this shard's parquet
        writer.flush_shard()                     # on disk before we move on

        # ---- 2. RECORD it as finished, so a re-run skips it ----
        done.add(name)
        rows_by_shard[name] = row_index
        if resume:
            # the offset is captured AFTER the flush, so it always marks a
            # boundary between whole shards
            save_progress(done, rows_by_shard, offset=writer.offset())

        # ---- 3. CLEAN the memory this shard used ----
        del pf, shard_records
        free_arrow_memory()

        mem = (f"   rss now {rss_mb():.0f} MB (peak {peak_mb():.0f})"
               if SHOW_MEMORY else "")
        print(f"  [{fi}/{len(files)}] {name}: {n_rows} rows, {n_cols} cols "
              f"-> saved, memory freed{mem}")

    writer.close()

    # ---------------- reports ----------------
    print(f"\nnormalized {total_rows} records to {len(CANONICAL_KEYS)} keys")
    if SHOW_MEMORY:
        print(f"memory: {rss_mb():.0f} MB now, {peak_mb():.0f} MB peak")

    if LOG_RENAMES and rename_counter:
        print("\nrenamed columns (original -> canonical):")
        for (src, dst), n in rename_counter.most_common():
            print(f"  {src!r} -> {dst}  ({n} rows)")

    if unmatched_counter:
        print("\ncolumns that matched nothing canonical"
              " (add real ones to MANUAL_ALIASES):")
        for key, n in unmatched_counter.most_common():
            print(f"  {key!r}: {n} rows")

    if missing_counter:
        print("\nkeys left as None — top 10:")
        for key, n in missing_counter.most_common(10):
            print(f"  {key}: None in {n}/{total_rows} records")

    if first_example:
        print("\n--- one example (long strings truncated for display) ---")
        for k, v in first_example.items():
            print(f"  {k}: {v!r}")

    return kept


def store(records=None, out_folder=OUT_PATH, mode=OUT_MODE):
    """Kept for compatibility -- writing now happens during the stream."""
    return


if __name__ == "__main__":
    read_parquet_folder()