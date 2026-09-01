"""
Record schema + pure-Python parsing/wiring for the paper knowledge graph.

=============================================================================
WHY THIS MODULE EXISTS
=============================================================================
Building the KG is two independent halves:

  PASS 1  raw *.json  ->  typed node TEXTS + typed EDGE index lists   (this file)
  PASS 2  texts       ->  MiniLM embeddings -> HeteroData tensors     (paper_graph.py)

Pass 1 is stdlib-only: no torch, no sentence-transformers, no GPU. Keeping it
here means the sample builder / inspector / visualiser can reproduce the EXACT
node and edge sets of the full graph on a laptop, and there is only ONE copy of
the schema logic to keep in sync.

`paper_graph.build_hetero_graph()` imports everything below, so the two paths
can never drift apart.
"""

import os
import glob
import json


# The reasoning aspects that become prediction-target heads (Option B).
ASPECTS = ("claim", "method", "result")

# Summary-level fields (one value per paper) -> aspect type.
SUMMARY_FIELD_TO_TYPE = {
    "methodological_details":       "method",
    "procedures_and_architectures": "method",   # format A/B
    "procedures_architectures":     "method",    # format C
    "key_results":                  "result",
}

# Claim-object sub-fields that hold the primary claim text.
CLAIM_TEXT_FIELDS = ("details", "description")   # A/B: 'details', C: 'description'

# NOTE: `contradicting_evidence` is deliberately NOT read. Evidence nodes come
# from `supporting_evidence` only — see EDGE_TYPES below.

# Node types in the graph, in the order they are reported.
NODE_TYPES = ("paper", "field", "claim", "method", "result", "evidence", "implication")

# (src, relation, dst) triples the builder can emit, and their RDF property name.
#
# The four reasoning relations form ONE chain per paper:
#
#     method --produces--> result --grounds--> evidence --supports--> claim
#                                                                       |
#                                                            implies    v
#                                                              implication
#
# Evidence sits BETWEEN the result and the claim it bears on, rather than
# hanging off the claim as a leaf. Direction is "what licenses what": a result
# grounds the evidence, and the evidence is what supports the claim. This is
# why the relation is named `supports` and not `supported_by` — the claim is
# the DESTINATION now, so the passive name would read backwards.
#
# There is no `challenges` counterpart: `contradicting_evidence` is not read
# into the graph at all, so every evidence node is supporting evidence and the
# chain has exactly one path through it.
EDGE_TYPES = (
    ("paper",    "has_claim",  "claim",       "hasClaim"),
    ("paper",    "has_method", "method",      "hasMethod"),
    ("paper",    "has_result", "result",      "hasResult"),
    ("paper",    "in_field",   "field",       "inField"),
    ("paper",    "cites",      "paper",       "cites"),
    ("method",   "produces",   "result",      "produces"),
    ("result",   "grounds",    "evidence",    "grounds"),
    ("evidence", "supports",   "claim",       "supports"),
    ("claim",    "implies",    "implication", "implies"),
)


# ===========================================================================
#  Record parsing helpers (handle all formats)
# ===========================================================================
def _coerce_summary(paper):
    """
    Return (summary_dict, field_label) for any supported record format, or
    (None, None) if the record is unusable (e.g. NON_SCIENTIFIC_TEXT with
    summary == null, or missing a field label).

    Format C (flat) : the summary fields sit at the top level of `paper`.
    Format A/B      : fields are nested under paper['summarization']['summary'].
    """
    # ---- Format C (flat): summary fields at the top level ----
    if "field_subfield" in paper and "summarization" not in paper:
        label = paper.get("field_subfield")
        return (paper, label) if label else (None, None)

    # ---- Format A/B: nested under 'summarization' ----
    raw = paper.get("summarization")
    if raw is None:
        return None, None
    if isinstance(raw, str):
        try:
            outer = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
    elif isinstance(raw, dict):
        outer = raw
    else:
        return None, None

    summary = outer.get("summary")
    if not summary or not isinstance(summary, dict):   # NON_SCIENTIFIC_TEXT -> null
        return None, None
    label = summary.get("field_subfield")
    return (summary, label) if label else (None, None)


def _coarse(label):
    """Collapse 'Field — Subfield (extra)' to just the coarse 'Field'."""
    for sep in (" — ", " - ", "—", " – "):
        if sep in label:
            return label.split(sep)[0].strip()
    return label.strip()


def _paper_id(paper):
    """Stable OpenAlex id for citation linking. Returns 'W123...' or None."""
    oid = paper.get("openalex_id") or paper.get("oa_doi")
    if not oid:
        return None
    return oid.rstrip("/").split("/")[-1].strip()


def _loads_deep(value, max_depth=4):
    """
    Decode a value that may be JSON encoded MORE THAN ONCE.

    The arXiv corpus stores `claims` double-encoded — a JSON string whose
    content is itself a JSON string holding the list. A single json.loads
    returns a str, and a naive isinstance(..., list) check then silently
    yields "no claims", wiping out every claim/evidence/implication node.
    So keep decoding while the result is still a string.
    """
    for _ in range(max_depth):
        if not isinstance(value, str):
            return value
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return value if not isinstance(value, str) else None


def _year(paper):
    """
    Publication year as an int, or None.

    Only used to sanity-check citation DIRECTION: a `cites` edge should run from
    the newer paper to the older one. Without this there is nothing to check the
    orientation against, and a silently reversed edge list looks exactly like a
    correct one.
    """
    for key in ("oa_year", "source_year", "year"):
        v = paper.get(key)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str) and v.strip()[:4].isdigit():
            return int(v.strip()[:4])
    return None


def _referenced_ids(paper):
    """Parse oa_referenced_works (JSON-string list of URLs) -> ['W...', ...]."""
    raw = paper.get("oa_referenced_works")
    if not raw:
        return []
    raw = _loads_deep(raw)
    if not isinstance(raw, list):
        return []
    return [u.rstrip("/").split("/")[-1].strip()
            for u in raw if isinstance(u, str) and u.strip()]


def _pick(summary, paper, *keys):
    """Return the first non-empty string among `keys`, searching summary then
    paper. Joins list-valued fields with spaces."""
    for src in (summary, paper):
        if not isinstance(src, dict):
            continue
        for k in keys:
            v = src.get(k)
            if isinstance(v, list):
                v = " ".join(str(x) for x in v if x)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _paper_text(summary, paper):
    """Text used to embed a paper node (title + executive summary)."""
    title = _pick(summary, paper, "title", "summary_title", "paper_title")
    exec_s = _pick(summary, paper, "executive_summary")
    txt = (title + ". " + exec_s).strip(" .")
    return txt if txt else None


def _get_claims_list(summary, paper):
    """Locate the list of claim objects (may be stringified JSON)."""
    for src in (summary, paper):
        if not isinstance(src, dict):
            continue
        cl = _loads_deep(src.get("claims"))     # may be encoded twice — see _loads_deep
        if isinstance(cl, list) and cl:
            return cl
    return []


def _claim_records(summary, paper):
    """
    Return one dict per claim:
        {text, supporting[list[str]], implications[list[str]]}
    Empty sub-fields become empty lists. Plain-string claims are also accepted.

    `contradicting_evidence` is read from neither format: the graph carries only
    supporting evidence, so there is exactly one evidence polarity and one path
    result -> evidence -> claim.
    """
    out = []
    for c in _get_claims_list(summary, paper):
        if isinstance(c, str):
            if c.strip():
                out.append(dict(text=c.strip(), supporting=[], implications=[]))
            continue
        if not isinstance(c, dict):
            continue

        text = ""
        for k in CLAIM_TEXT_FIELDS:
            v = c.get(k)
            if isinstance(v, str) and v.strip():
                text = v.strip()
                break
        if not text:
            continue

        def as_list(key):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return [v.strip()]
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            return []

        out.append(dict(
            text=text,
            supporting=as_list("supporting_evidence"),
            implications=as_list("implications"),
        ))
    return out


def _aspect_text(summary, paper, aspect):
    """Single text (or None) for a summary-level aspect (method / result)."""
    for field, atype in SUMMARY_FIELD_TO_TYPE.items():
        if atype != aspect:
            continue
        t = _pick(summary, paper, field)
        if t:
            return t
    return None


def _iter_raw_records(raw_dir, prefix=None):
    """
    Yield every record dict from all *.json files (dicts or lists thereof).

    `raw_dir` may be either a DIRECTORY of .json files or a .ZIP ARCHIVE of
    them. Reading the zip directly matters at corpus scale: the arXiv release
    ships as one 6 GB zip of 208,982 files, and extracting a single domain
    would duplicate gigabytes on disk for no benefit.

    `prefix` selects a subtree inside the archive — the arXiv zip is laid out as
    `<domain>/<field>/<paper>.json`, so prefix='physical_sciences/' is a cheap,
    exact domain filter that never opens an out-of-scope record. It may also be a
    LIST of prefixes, which is how a graph spanning two fields (e.g.
    materials_science + chemical_engineering) is built without walking the other
    200k records just to throw them away.

    Members are visited in sorted order so node rows stay deterministic.
    """
    if isinstance(prefix, str):
        prefix = (prefix,)
    elif prefix is not None:
        prefix = tuple(prefix)
        if not prefix:
            prefix = None

    def _emit(data, where):
        if isinstance(data, dict):
            yield data
        elif isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    yield rec

    if os.path.isfile(raw_dir) and raw_dir.lower().endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(raw_dir) as zf:
            names = sorted(n for n in zf.namelist()
                           if n.endswith(".json")
                           and (prefix is None or n.startswith(prefix)))
            for name in names:
                try:
                    data = json.loads(zf.read(name).decode("utf-8"))
                except Exception as ex:
                    print(f"[paper_graph] WARNING: skip {name}: {ex}")
                    continue
                yield from _emit(data, name)
        return

    for fp in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        if prefix is not None and not os.path.basename(fp).startswith(prefix):
            continue
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as ex:
            print(f"[paper_graph] WARNING: skip {fp}: {ex}")
            continue
        yield from _emit(data, fp)


# ===========================================================================
#  PASS 1 — parse raw records into per-paper dicts (strings only, cheap)
# ===========================================================================
def parse_records(raw_dir, coarse_label=True, limit=None, progress=True,
                  field=None, domain=None, label_source="field_subfield",
                  prefix=None, paper_ids=None):
    """
    Read every raw record under `raw_dir` and return (papers, skipped).

    Each element of `papers` is::

        {pid, text, claims[{text, supporting, implications}],
         method, result, field, refs}

    A record is SKIPPED when it has no usable summary/field label (typically a
    NON_SCIENTIFIC_TEXT record whose `summary` is null) or no title +
    executive_summary to embed as the paper node.

    Subsetting — build a KG for one slice of the corpus rather than all of it:
      field  : keep only these coarse field labels ('Medicine', 'Biology', …),
               matched case-insensitively against the same label that becomes
               the paper's `field` hub node.
      domain : keep only these OpenAlex top-level domains ('Health Sciences',
               'Physical Sciences', …), read from the record's `domain` key.

    `limit` is applied AFTER filtering, so `--field Medicine --limit 500` gives
    500 Medicine papers rather than 500 records of which some are Medicine.

    label_source — which record key becomes the paper's `field` hub node:
      'field_subfield' (default) is the LLM's own label, prose-shaped
          ('Medicine — Pulmonology'), and gets coarsened to 'Medicine'.
      'field' / 'subfield' are OpenAlex's classification.
    These disagree, and which is right depends on the corpus. In the arXiv
    corpus `field_subfield` holds arXiv CATEGORY CODES, so it shatters into
    dozens of hubs ('Computer Science' for a dentistry paper); there,
    label_source='field' is the correct choice. In the PubMed corpus
    `field_subfield` is proper prose and is the better label.
    """
    if progress:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **k):
                return x
    else:
        def tqdm(x, **k):
            return x

    want_field = {f.strip().lower() for f in field} if field else None
    want_domain = {d.strip().lower() for d in domain} if domain else None
    # exact OpenAlex ids ('W123...'), for pulling one specific subgraph out of a
    # multi-million-node graph — e.g. a cites-connected pair worth visualising.
    want_pids = {p.strip().rstrip("/").split("/")[-1] for p in paper_ids} if paper_ids else None

    # what the corpus actually offers, so a filter that matches nothing can say why
    seen_fields, seen_domains = set(), set()

    papers, skipped, filtered = [], 0, 0
    for rec in tqdm(_iter_raw_records(raw_dir, prefix=prefix), desc="Parsing"):
        if limit is not None and len(papers) >= limit:
            break
        summary, label = _coerce_summary(rec)
        if summary is None:
            skipped += 1
            continue
        if label_source != "field_subfield":
            alt = rec.get(label_source)
            if not isinstance(alt, str) or not alt.strip():
                skipped += 1
                continue
            label = alt.strip()
        ptext = _paper_text(summary, rec)
        if ptext is None:
            skipped += 1
            continue

        # ---- subset filters (applied before `limit` is counted) ----
        if want_pids is not None and _paper_id(rec) not in want_pids:
            filtered += 1
            continue

        # `seen_*` only feed the "your filter matched nothing, here is what was
        # actually present" message, so they are recorded only when a filter is
        # active — otherwise every record paid for a set insert and a second
        # _coarse() call whose result is thrown away.
        coarse = _coarse(label)
        dom = rec.get("domain")
        if want_domain is not None or want_field is not None:
            seen_domains.add(dom.strip() if isinstance(dom, str) and dom.strip()
                             else "(no 'domain' key)")
            seen_fields.add(coarse)

        if want_domain is not None:
            if not isinstance(dom, str) or dom.strip().lower() not in want_domain:
                filtered += 1
                continue
        if want_field is not None:
            if coarse.lower() not in want_field:
                filtered += 1
                continue

        papers.append(dict(
            pid=_paper_id(rec),
            text=ptext,
            claims=_claim_records(summary, rec),
            method=_aspect_text(summary, rec, "method"),
            result=_aspect_text(summary, rec, "result"),
            field=coarse if coarse_label else label.strip(),
            refs=_referenced_ids(rec),
            year=_year(rec),
        ))
    if filtered:
        print(f"[schema] subset filter dropped {filtered} out-of-scope papers "
              f"(field={sorted(want_field) if want_field else 'any'}, "
              f"domain={sorted(want_domain) if want_domain else 'any'})")
    if not papers and filtered:
        # the filter is the reason nothing came through — say what WAS available
        if want_domain is not None:
            print(f"[schema] domains present in this corpus: {sorted(seen_domains)}")
            if seen_domains == {"(no 'domain' key)"}:
                print("[schema] these records carry no OpenAlex 'domain' field — "
                      "use --field instead, which reads 'field_subfield'.")
        if want_field is not None:
            print(f"[schema] fields present in this corpus: {sorted(seen_fields)}")
    return papers, skipped


def coverage_report(papers, skipped):
    """
    Counts printed by the builder so you can see whether the schema matched.

    `claims_without_evidence` is the one to watch. A claim now reaches the rest
    of its paper's chain only THROUGH its evidence (result -> evidence -> claim),
    so a claim with no supporting_evidence hangs off `has_claim` alone. It is 0
    on the corpora built so far; if it ever climbs, the chain is not the shape
    the schema advertises.
    """
    return dict(
        num_papers=len(papers),
        skipped=skipped,
        papers_with_claims=sum(1 for p in papers if p["claims"]),
        papers_with_method=sum(1 for p in papers if p["method"]),
        papers_with_result=sum(1 for p in papers if p["result"]),
        total_claims=sum(len(p["claims"]) for p in papers),
        total_evidence=sum(len(c["supporting"])
                           for p in papers for c in p["claims"]),
        total_implications=sum(len(c["implications"])
                               for p in papers for c in p["claims"]),
        claims_without_evidence=sum(1 for p in papers for c in p["claims"]
                                    if not c["supporting"]),
    )


# ===========================================================================
#  PASS 1b — flatten papers into node tables + typed edge lists
# ===========================================================================
def build_tables(papers):
    """
    Turn parsed papers into the flat node/edge tables the HeteroData is made of.

    Returns a dict with:
        texts  : {node_type: [text, ...]}          row i == node i of that type
        owner  : {node_type: [paper_row, ...]}     which paper each node came from
        edges  : {(src, rel, dst): ([src_rows], [dst_rows])}
        field_map   : field string -> field node row
        paper_row   : OpenAlex id  -> paper node row
        n_dangling  : citations pointing outside the corpus (dropped)

    Node rows are assigned in corpus order, so this is deterministic given the
    same set of input files.
    """
    # ---------------- id maps ----------------
    paper_row = {}
    for i, p in enumerate(papers):
        if p["pid"] and p["pid"] not in paper_row:
            paper_row[p["pid"]] = i
    fields_sorted = sorted({p["field"] for p in papers})
    field_map = {f: i for i, f in enumerate(fields_sorted)}

    # ---------------- flatten nodes + intra-paper edges ----------------
    # Evidence is a LINK in the chain, not a leaf: an evidence node's edge runs
    # evidence -> claim, so the claim is the destination and `sup_src` holds the
    # evidence row. (Before, this ran claim -> evidence.)
    claim_texts, claim_owner = [], []
    evid_texts, evid_owner = [], []
    sup_src, sup_dst = [], []                    # (evidence, supports, claim)
    impl_texts, impl_owner = [], []
    impl_src, impl_dst = [], []                  # (claim, implies, implication)

    for i, p in enumerate(papers):
        for c in p["claims"]:
            cidx = len(claim_texts)
            claim_texts.append(c["text"])
            claim_owner.append(i)
            for e in c["supporting"]:
                sup_src.append(len(evid_texts))
                sup_dst.append(cidx)
                evid_texts.append(e)
                evid_owner.append(i)
            for im in c["implications"]:
                impl_src.append(cidx)
                impl_dst.append(len(impl_texts))
                impl_texts.append(im)
                impl_owner.append(i)

    # method / result (one per paper that has it)
    method_texts, method_owner = [], []
    result_texts, result_owner = [], []
    for i, p in enumerate(papers):
        if p["method"]:
            method_owner.append(i)
            method_texts.append(p["method"])
        if p["result"]:
            result_owner.append(i)
            result_texts.append(p["result"])

    # intra-paper structural edges: method->produces->result, result->grounds->evidence.
    # Both are IMPOSED — the LLM summary never says which evidence came from
    # which result, so a paper's single result is wired to all of its evidence.
    # `grounds` lands on EVIDENCE (not, as before, straight onto the claims):
    # that is what turns the paper into the longer chain
    # method -> result -> evidence -> claim -> implication.
    method_row = {mo: k for k, mo in enumerate(method_owner)}   # paper idx -> method node idx
    result_row = {ro: k for k, ro in enumerate(result_owner)}   # paper idx -> result node idx
    prod_src, prod_dst = [], []                                 # (method, produces, result)
    for pi in range(len(papers)):
        if pi in method_row and pi in result_row:
            prod_src.append(method_row[pi])
            prod_dst.append(result_row[pi])
    grnd_src, grnd_dst = [], []                                 # (result, grounds, evidence)
    for eidx, owner in enumerate(evid_owner):
        if owner in result_row:
            grnd_src.append(result_row[owner])
            grnd_dst.append(eidx)

    # ---------------- paper-level edges ----------------
    inf_src = list(range(len(papers)))
    inf_dst = [field_map[p["field"]] for p in papers]

    # ---- (paper, cites, paper): a DIRECTED citation graph ----
    # Orientation is citing -> cited: edge (i, j) means paper i lists paper j in
    # its own reference list. Two kinds of junk are removed so that orientation
    # actually means something:
    #   self-loops  — some records list their own openalex_id in
    #                 oa_referenced_works. An i->i edge has no direction, and in
    #                 the health-sciences slice 91 of 147 cites edges were these.
    #   duplicates  — the same reference can appear twice in one record, which
    #                 would otherwise weight that pair twice during aggregation.
    # Everything pointing outside the corpus stays dropped as dangling.
    cites_src, cites_dst = [], []
    n_dangling = n_self_cites = n_dup_cites = 0
    seen_pairs = set()
    for i, p in enumerate(papers):
        for r in p["refs"]:
            j = paper_row.get(r)
            if j is None:
                n_dangling += 1
                continue
            if j == i:
                n_self_cites += 1
                continue
            if (i, j) in seen_pairs:
                n_dup_cites += 1
                continue
            seen_pairs.add((i, j))
            cites_src.append(i)
            cites_dst.append(j)

    def owner_edge(owner_list):
        """(owner_paper_row -> sequential_node_row) edge pair."""
        return list(owner_list), list(range(len(owner_list)))

    edges = {
        ("paper",    "has_claim",  "claim"):       owner_edge(claim_owner),
        ("paper",    "has_method", "method"):      owner_edge(method_owner),
        ("paper",    "has_result", "result"):      owner_edge(result_owner),
        ("paper",    "in_field",   "field"):       (inf_src, inf_dst),
        ("paper",    "cites",      "paper"):       (cites_src, cites_dst),
        ("method",   "produces",   "result"):      (prod_src, prod_dst),
        ("result",   "grounds",    "evidence"):    (grnd_src, grnd_dst),
        ("evidence", "supports",   "claim"):       (sup_src, sup_dst),
        ("claim",    "implies",    "implication"): (impl_src, impl_dst),
    }

    return dict(
        texts={
            "paper":       [p["text"] for p in papers],
            "field":       fields_sorted,
            "claim":       claim_texts,
            "method":      method_texts,
            "result":      result_texts,
            "evidence":    evid_texts,
            "implication": impl_texts,
        },
        owner={
            "paper":       list(range(len(papers))),
            "field":       [None] * len(fields_sorted),   # shared hubs, no single owner
            "claim":       claim_owner,
            "method":      method_owner,
            "result":      result_owner,
            "evidence":    evid_owner,
            "implication": impl_owner,
        },
        edges=edges,
        field_map=field_map,
        paper_row=paper_row,
        field_of_paper=inf_dst,
        n_dangling=n_dangling,
        n_self_cites=n_self_cites,
        n_dup_cites=n_dup_cites,
    )


def citation_report(papers, tables):
    """
    Describe the directed citation graph: what was dropped, how lopsided the
    degree distribution is, and whether the edges actually point backwards in
    time (newer -> older).

    `edges_backwards_in_time` is the correctness check. If it were large
    relative to `edges_with_years`, the src/dst orientation would be reversed
    somewhere; a handful is normal (preprints, OpenAlex year errors).
    """
    src, dst = tables["edges"][("paper", "cites", "paper")]
    n = len(src)
    out_deg, in_deg = {}, {}
    for s, d in zip(src, dst):
        out_deg[s] = out_deg.get(s, 0) + 1
        in_deg[d] = in_deg.get(d, 0) + 1

    dated = backwards = 0
    for s, d in zip(src, dst):
        ys, yd = papers[s].get("year"), papers[d].get("year")
        if ys is None or yd is None:
            continue
        dated += 1
        if ys < yd:                      # citing paper older than the one it cites
            backwards += 1

    return dict(
        directed=True,
        orientation="src=citing paper, dst=cited paper",
        edges=n,
        dangling_dropped=tables["n_dangling"],
        self_citations_dropped=tables["n_self_cites"],
        duplicate_edges_dropped=tables["n_dup_cites"],
        papers_citing=len(out_deg),
        papers_cited=len(in_deg),
        max_out_degree=max(out_deg.values()) if out_deg else 0,
        max_in_degree=max(in_deg.values()) if in_deg else 0,
        edges_with_years=dated,
        edges_backwards_in_time=backwards,
    )
