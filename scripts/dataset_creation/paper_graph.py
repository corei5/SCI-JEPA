"""
Part 2: Heterogeneous paper reasoning graph (scope A) — HIERARCHICAL SUBGRAPHS.

=============================================================================
WHAT THIS BUILDS
=============================================================================
ONE big HeteroData graph in which EACH PAPER is an internal reasoning
SUBGRAPH. Instead of a flat "paper -> aspect" star, a paper's aspects are
wired into a single CHAIN, mirroring scientific structure:

        method --produces--> result --grounds--> evidence
        evidence --supports--> claim --implies--> implication

i.e. paper -> method -> result -> evidence -> claim -> implication.

Evidence is a link in that chain, not a leaf hanging off a claim: a result
grounds the evidence, and the evidence is what supports the claim. Only
SUPPORTING evidence is represented — `contradicting_evidence` is not read into
the graph, so there is exactly one polarity and one path through each paper.

All paper subgraphs share ONE graph and are connected to each other via
`in_field` (shared field hubs) and `cites` (intra-corpus citations).

Why a single big graph (not 57k separate subgraphs)?
    With PyG `to_hetero`, message passing is already LOCAL: a node only
    aggregates from its true neighbours. So typed intra-paper edges give the
    exact "subgraph" behaviour while keeping one clean HeteroData that plugs
    straight into the masked-target reasoning task. No hierarchical pooling
    needed, and the JEPA eval stays intact.

=============================================================================
NODES
=============================================================================
    paper       : MiniLM(title + executive_summary)          [1 / paper]
    claim       : MiniLM(claim.details | claim.description)   [many / paper]
    method      : MiniLM(methodological_details | procedures) [~1 / paper]
    result      : MiniLM(key_results)                         [~1 / paper]
    evidence    : MiniLM(claim.supporting_evidence)           [many / claim]
    implication : MiniLM(claim.implications)                  [many / claim]
    field       : MiniLM(field_subfield string)               [shared hubs]

=============================================================================
EDGES
=============================================================================
  Paper-level (connect a paper to its aspects and to other papers):
    (paper,  has_claim,      claim)
    (paper,  has_method,     method)
    (paper,  has_result,     result)
    (paper,  in_field,       field)
    (paper,  cites,          paper)     # intra-corpus only; dangling dropped

  Intra-paper reasoning chain (aspects wired to each other, in order):
    (method,   produces,  result)       # imposed method->result structure
    (result,   grounds,   evidence)     # imposed result->evidence structure
    (evidence, supports,  claim)        # from supporting_evidence
    (claim,    implies,   implication)

=============================================================================
RECORD FORMATS HANDLED
=============================================================================
  Format A/B : record has 'summarization' (str or dict) -> {'summary': {...}};
               claims use key 'details'.
  Format C   : summary fields live at the TOP LEVEL of the record (flat);
               claims use key 'description'.
  Skipped    : NON_SCIENTIFIC_TEXT records where summary == null, and any
               record missing a usable title/executive_summary or field label.
  PARTIAL_SCIENTIFIC_TEXT records are KEPT (they still carry a valid summary).

Run once to build+cache; the resulting .pt is reused afterwards. Set
REBUILD_HETERO=1 (see paper_reason.py) to force a rebuild when the schema
changes.

=============================================================================
WHERE THE CODE LIVES
=============================================================================
Parsing and edge wiring (PASS 1) live in `schema.py` — stdlib only, so the
sample builder and the visualiser reproduce the exact same nodes and edges
without torch. This file is PASS 2: embed the texts and pack the tables into
a HeteroData.
"""

import os
import gc
import torch
from torch_geometric.data import HeteroData

from .schema import (          # noqa: F401  (re-exported for existing importers)
    ASPECTS,
    SUMMARY_FIELD_TO_TYPE,
    CLAIM_TEXT_FIELDS,
    NODE_TYPES,
    EDGE_TYPES,
    build_tables,
    citation_report,
    coverage_report,
    parse_records,
    _aspect_text,
    _claim_records,
    _coarse,
    _coerce_summary,
    _get_claims_list,
    _iter_raw_records,
    _paper_id,
    _paper_text,
    _pick,
    _referenced_ids,
)


# ===========================================================================
#  Build
# ===========================================================================
def build_hetero_graph(raw_dir, cache_path,
                       embed_model="all-MiniLM-L6-v2",
                       coarse_label=True,
                       rebuild=False):
    """
    Build (or load cached) the hierarchical multi-aspect HeteroData.

    Parameters
    ----------
    raw_dir      : directory of raw *.json paper records.
    cache_path   : path to save/load the processed .pt graph.
    embed_model  : SentenceTransformer model name for node embeddings.
    coarse_label : collapse 'Field — Subfield' to coarse 'Field' for the field node.
    rebuild      : if True, ignore any cache and rebuild from scratch.

    Returns
    -------
    (data, meta) : HeteroData and a dict of counts / id maps.
    """
    if (not rebuild) and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        print(f"[paper_graph] loaded cache {cache_path}")
        return blob["data"], blob["meta"]

    from sentence_transformers import SentenceTransformer

    # ---------------- PASS 1: parse (strings only, cheap) ----------------
    papers, skipped = parse_records(raw_dir, coarse_label=coarse_label)
    if not papers:
        raise RuntimeError("No usable papers parsed. Check raw_dir / schema.")

    # ---------------- diagnostic coverage report ----------------
    cov = coverage_report(papers, skipped)
    print(f"[paper_graph] papers={cov['num_papers']} skipped={cov['skipped']}")
    print(f"[paper_graph] coverage -> "
          f"claims:{cov['papers_with_claims']}/{cov['num_papers']} ({cov['total_claims']}) | "
          f"method:{cov['papers_with_method']} | result:{cov['papers_with_result']} | "
          f"evidence:{cov['total_evidence']} | implications:{cov['total_implications']}",
          flush=True)
    for name, n in [("claim", cov["papers_with_claims"]),
                    ("method", cov["papers_with_method"]),
                    ("result", cov["papers_with_result"])]:
        if n == 0:
            print(f"[paper_graph] ★ WARNING: 0 '{name}' found — check field names/format.")
    if cov["total_evidence"] == 0:
        print("[paper_graph] ★ WARNING: 0 evidence nodes — supporting_evidence"
              " missing; the chain breaks at result and no claim is reachable"
              " from it.")
    elif cov["claims_without_evidence"]:
        # Evidence is now the ONLY route from a result to a claim, so a claim
        # with none is attached by `has_claim` alone. Worth saying out loud.
        print(f"[paper_graph] note: {cov['claims_without_evidence']}/"
              f"{cov['total_claims']} claims have no supporting evidence and so "
              f"sit off the result->evidence->claim chain.")

    # ---------------- flatten nodes + typed edges ----------------
    tables = build_tables(papers)
    texts, edges = tables["texts"], tables["edges"]

    cr = citation_report(papers, tables)
    n_cites = cr["edges"]
    print(f"[paper_graph] cites (directed, {cr['orientation']}): {n_cites} edges | "
          f"dropped {cr['dangling_dropped']} dangling / "
          f"{cr['self_citations_dropped']} self-citations / "
          f"{cr['duplicate_edges_dropped']} duplicates", flush=True)
    print(f"[paper_graph]   {cr['papers_citing']} citing / {cr['papers_cited']} cited | "
          f"max out-degree={cr['max_out_degree']} in-degree={cr['max_in_degree']} | "
          f"{cr['edges_backwards_in_time']}/{cr['edges_with_years']} dated edges "
          f"cite a NEWER paper (should be near zero)", flush=True)
    if not n_cites:
        print("[paper_graph] ★ WARNING: NO intra-corpus citation edges. "
              "Reasoning relies on subgraph + field structure only.")

    # ---------------- PASS 2: embed ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(embed_model, device=device)

    # paper embeddings first (defines emb_dim used by the empty-tensor fallback)
    paper_x = torch.tensor(
        model.encode(texts["paper"], batch_size=256,
                     convert_to_numpy=True, show_progress_bar=True),
        dtype=torch.float,
    )
    emb_dim = paper_x.size(1)

    def embed(node_texts, prog=False):
        if not node_texts:
            return torch.empty((0, emb_dim), dtype=torch.float)
        arr = model.encode(node_texts, batch_size=256, convert_to_numpy=True,
                           show_progress_bar=prog)
        return torch.tensor(arr, dtype=torch.float)

    node_x = {"paper": paper_x}
    for nt in ("field", "claim", "method", "result", "evidence", "implication"):
        node_x[nt] = embed(texts[nt], prog=(nt != "field"))

    del model
    gc.collect()

    # ---------------- assemble HeteroData ----------------
    def edge_tensor(pair):
        src, dst = pair
        s = torch.tensor(src, dtype=torch.long) if src else torch.empty(0, dtype=torch.long)
        d = torch.tensor(dst, dtype=torch.long) if dst else torch.empty(0, dtype=torch.long)
        return torch.stack([s, d], 0)

    data = HeteroData()
    for nt in NODE_TYPES:
        data[nt].x = node_x[nt]
    # convenience field label per paper
    data["paper"].field_y = torch.tensor(tables["field_of_paper"], dtype=torch.long)

    for et, pair in edges.items():
        data[et].edge_index = edge_tensor(pair)

    meta = dict(
        field_map=tables["field_map"],
        aspects=list(ASPECTS),
        num_papers=len(papers),
        num_claims=len(texts["claim"]),
        num_methods=len(texts["method"]),
        num_results=len(texts["result"]),
        num_evidence=len(texts["evidence"]),
        num_implications=len(texts["implication"]),
        num_fields=len(texts["field"]),
        emb_dim=emb_dim,
        skipped=skipped,
        citations=cr,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"data": data, "meta": meta}, cache_path)
    print(f"[paper_graph] saved cache -> {cache_path}")
    print(f"[paper_graph] nodes: paper={meta['num_papers']} claim={meta['num_claims']} "
          f"method={meta['num_methods']} result={meta['num_results']} "
          f"evidence={meta['num_evidence']} impl={meta['num_implications']} "
          f"field={meta['num_fields']} | emb_dim={meta['emb_dim']}")
    return data, meta
