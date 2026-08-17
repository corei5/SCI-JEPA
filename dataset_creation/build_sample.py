"""
Build a SMALL, inspectable sample of the paper knowledge graph.

This runs PASS 1 only (parse -> nodes -> typed edges) and skips the MiniLM
embedding pass, so it needs nothing but the standard library and finishes
instantly. The nodes and edges it emits are byte-for-byte the same ones
`paper_graph.build_hetero_graph()` would embed — both call `schema.build_tables`.

Outputs (default: dataset_creation/sample/):

    nodes.json        flat node table   [{id, type, local_idx, owner, text}]
    edges.json        flat edge table   [{src, rel, dst}]
    subgraphs.jsonl   one nested JSON reasoning subgraph per paper
    sample_kg.ttl     the same graph as RDF triples (matches export_kg/schema.ttl)
    stats.json        node/edge counts + coverage report

Usage
-----
    python -m dataset_creation.build_sample                      # sample_raw/
    python -m dataset_creation.build_sample --raw-dir Analysis/dataset/papers/raw \
                                            --limit 5 --out /tmp/kg_sample
"""

import argparse
import json
import os

from .schema import (EDGE_TYPES, NODE_TYPES, build_tables, citation_report,
                     coverage_report, parse_records)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW = os.path.join(HERE, "sample_raw")
DEFAULT_OUT = os.path.join(HERE, "sample")

RDF_PROP = {(s, r, d): prop for s, r, d, prop in EDGE_TYPES}
RDF_CLASS = {nt: nt[:1].upper() + nt[1:] for nt in NODE_TYPES}


def node_id(ntype, idx):
    """Readable, stable node id: 'claim_12'."""
    return f"{ntype}_{idx}"


def build_sample(raw_dir, out_dir, limit=None, coarse_label=True, snippet=None,
                 field=None, domain=None, label_source="field_subfield",
                 prefix=None, paper_ids=None):
    papers, skipped = parse_records(raw_dir, coarse_label=coarse_label,
                                    limit=limit, progress=False,
                                    field=field, domain=domain,
                                    label_source=label_source, prefix=prefix,
                                    paper_ids=paper_ids)
    if not papers:
        scope = f" matching field={field} domain={domain}" if (field or domain) else ""
        raise RuntimeError(f"No usable papers parsed from {raw_dir}{scope}.")

    cov = coverage_report(papers, skipped)
    tables = build_tables(papers)
    texts, owner, edges = tables["texts"], tables["owner"], tables["edges"]

    os.makedirs(out_dir, exist_ok=True)

    def cut(t):
        if snippet and len(t) > snippet:
            return t[:snippet].rstrip() + "…"
        return t

    # ---------------- flat node / edge tables ----------------
    # Generators, not lists: at corpus scale these tables are the whole graph
    # (the physical-sciences slice is 1.1M nodes), and materialising them would
    # hold a second full copy of every node's text in RAM purely to hand it to
    # json.dumps. Each is consumed twice — once for the .json, once for the TTL —
    # so they are functions returning a fresh generator rather than generators.
    def iter_nodes():
        for ntype in NODE_TYPES:
            owners = owner[ntype]
            for i, t in enumerate(texts[ntype]):
                yield dict(id=node_id(ntype, i), type=ntype, local_idx=i,
                           owner=owners[i], text=cut(t))

    def iter_edges():
        for (src_t, rel, dst_t), (srcs, dsts) in edges.items():
            for s, d in zip(srcs, dsts):
                yield dict(src=node_id(src_t, s), rel=rel,
                           dst=node_id(dst_t, d))

    n_nodes = sum(len(texts[nt]) for nt in NODE_TYPES)
    n_edges = sum(len(v[0]) for v in edges.values())

    # ---------------- per-paper nested subgraphs ----------------
    # claim -> its evidence / implications, so a subgraph reads top-down.
    claim_support, claim_contra, claim_impl = {}, {}, {}
    for key, bucket in ((("claim", "supported_by", "evidence"), claim_support),
                        (("claim", "challenged_by", "evidence"), claim_contra),
                        (("claim", "implies", "implication"), claim_impl)):
        for c, e in zip(*edges[key]):
            bucket.setdefault(c, []).append(e)

    method_of = {o: k for k, o in enumerate(owner["method"])}
    result_of = {o: k for k, o in enumerate(owner["result"])}
    claims_of = {}
    for c, o in enumerate(owner["claim"]):
        claims_of.setdefault(o, []).append(c)

    # `cites` is directed, so a paper has two distinct neighbourhoods. Both are
    # written out — cited_by is derived from the same edge list, not a new
    # relation, so the graph still has exactly one citation edge type.
    cites_out, cites_in = {}, {}
    for s, d in zip(*edges[("paper", "cites", "paper")]):
        cites_out.setdefault(s, []).append(d)
        cites_in.setdefault(d, []).append(s)

    subgraphs = []
    for pi, p in enumerate(papers):
        sg = dict(
            id=node_id("paper", pi),
            openalex_id=p["pid"],
            field=p["field"],
            text=cut(texts["paper"][pi]),
            method=(dict(id=node_id("method", method_of[pi]),
                         text=cut(texts["method"][method_of[pi]]))
                    if pi in method_of else None),
            result=(dict(id=node_id("result", result_of[pi]),
                         text=cut(texts["result"][result_of[pi]]))
                    if pi in result_of else None),
            claims=[
                dict(
                    id=node_id("claim", c),
                    text=cut(texts["claim"][c]),
                    supported_by=[dict(id=node_id("evidence", e),
                                       text=cut(texts["evidence"][e]))
                                  for e in claim_support.get(c, [])],
                    challenged_by=[dict(id=node_id("evidence", e),
                                        text=cut(texts["evidence"][e]))
                                   for e in claim_contra.get(c, [])],
                    implies=[dict(id=node_id("implication", m),
                                  text=cut(texts["implication"][m]))
                             for m in claim_impl.get(c, [])],
                )
                for c in claims_of.get(pi, [])
            ],
            year=p.get("year"),
            cites=[node_id("paper", d) for d in cites_out.get(pi, [])],
            cited_by=[node_id("paper", s) for s in cites_in.get(pi, [])],
        )
        subgraphs.append(sg)

    # ---------------- stats ----------------
    stats = dict(
        raw_dir=os.path.abspath(raw_dir),
        coarse_label=coarse_label,
        subset=dict(field=list(field) if field else None,
                    domain=list(domain) if domain else None,
                    limit=limit),
        label_source=label_source,
        prefix=prefix,
        paper_ids=list(paper_ids) if paper_ids else None,
        coverage=cov,
        nodes={nt: len(texts[nt]) for nt in NODE_TYPES},
        total_nodes=n_nodes,
        edges={f"{s}__{r}__{d}": len(v[0]) for (s, r, d), v in edges.items()},
        total_edges=n_edges,
        dangling_citations_dropped=tables["n_dangling"],
        citations=citation_report(papers, tables),
        fields=sorted(tables["field_map"]),
    )

    # ---------------- write ----------------
    def dump_array(name, records):
        """
        Write a JSON array with ONE RECORD PER LINE.

        `json.dump(..., indent=2)` silently disables json's C encoder — the whole
        array goes through the pure-Python _iterencode path, which was 37% of
        this script's runtime, and it padded every record with ~8 lines of
        indentation. Emitting each record with json.dumps() keeps the C encoder,
        produces the same valid JSON both readers already parse with json.load,
        and stays line-diffable.
        """
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write("[")
            first = True
            for rec in records:
                fh.write("\n" if first else ",\n")
                first = False
                fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("]\n" if first else "\n]\n")

    dump_array("nodes.json", iter_nodes())
    dump_array("edges.json", iter_edges())
    # stats is small and meant to be read by a human, so it keeps the indenting.
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "subgraphs.jsonl"), "w", encoding="utf-8") as fh:
        for sg in subgraphs:
            fh.write(json.dumps(sg, ensure_ascii=False) + "\n")
    write_ttl(os.path.join(out_dir, "sample_kg.ttl"), iter_nodes(), iter_edges(), edges)

    print(f"[build_sample] papers={cov['num_papers']} skipped={cov['skipped']}")
    for nt in NODE_TYPES:
        print(f"[build_sample]   {nt:12s} {len(texts[nt]):>5d} nodes")
    for (s, r, d), v in edges.items():
        print(f"[build_sample]   ({s}, {r}, {d}): {len(v[0])}")
    cr = stats["citations"]
    print(f"[build_sample] cites (directed, {cr['orientation']}): {cr['edges']} edges "
          f"| {cr['papers_citing']} citing / {cr['papers_cited']} cited "
          f"| max out={cr['max_out_degree']} in={cr['max_in_degree']}")
    print(f"[build_sample]   dropped: {cr['dangling_dropped']} dangling, "
          f"{cr['self_citations_dropped']} self-citations, "
          f"{cr['duplicate_edges_dropped']} duplicates")
    print(f"[build_sample]   direction check: {cr['edges_backwards_in_time']}"
          f"/{cr['edges_with_years']} dated edges cite a NEWER paper "
          f"(should be near zero if src->dst is citing->cited)")
    print(f"[build_sample] total {n_nodes} nodes / {n_edges} edges -> {out_dir}")
    # Only `stats` is returned. The node and edge tables ARE the graph, and no
    # caller ever used them (both readers load the written .json instead), so
    # returning them just pinned the whole corpus in memory after the write.
    return stats


def write_ttl(path, nodes, flat_edges, edges):
    """
    Emit the sample as RDF, using the vocabulary of export_kg/schema.ttl.

    `nodes` and `flat_edges` are iterated once each and streamed straight to
    disk. The previous version accumulated every triple in a list and joined it
    into one string, which for the physical-sciences slice meant building an
    866 MB string in memory before a single byte was written.
    """
    def esc(t):
        return (t.replace("\\", "\\\\").replace('"', '\\"')
                 .replace("\n", " ").replace("\r", " "))

    rel_prop = {r: RDF_PROP[(s, r, d)] for (s, r, d) in edges}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("@prefix ex: <http://example.org/kg#> .\n"
                 "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
                 "\n# Nodes\n")
        for n in nodes:
            fh.write(f'ex:{n["id"]} a ex:{RDF_CLASS[n["type"]]} ; '
                     f'rdfs:label "{esc(n["text"])}" .\n')
        fh.write("\n# Edges\n")
        for e in flat_edges:
            fh.write(f'ex:{e["src"]} ex:{rel_prop[e["rel"]]} ex:{e["dst"]} .\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=DEFAULT_RAW,
                    help="directory of raw paper *.json (default: sample_raw/)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output directory (default: sample/)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N usable papers (counted AFTER the subset filters)")
    ap.add_argument("--snippet", type=int, default=None,
                    help="truncate node texts to N chars in the output files")
    ap.add_argument("--full-label", action="store_true",
                    help="keep 'Field — Subfield' instead of collapsing to 'Field'")
    ap.add_argument("--domain", action="append", default=None, metavar="NAME",
                    help="build only this OpenAlex domain, e.g. --domain 'Health Sciences'. "
                         "Reads the record's top-level 'domain' key; records lacking it are "
                         "dropped. Repeatable.")
    ap.add_argument("--field", action="append", default=None, metavar="NAME",
                    help="build only this coarse field, e.g. --field Medicine. Matches the "
                         "same label that becomes the graph's `field` hub node. Repeatable.")
    ap.add_argument("--label-source", default="field_subfield",
                    choices=["field_subfield", "field", "subfield"],
                    help="which record key becomes the `field` hub node. Use 'field' for "
                         "the arXiv corpus, whose 'field_subfield' holds arXiv category "
                         "codes rather than prose (default: field_subfield)")
    ap.add_argument("--prefix", action="append", default=None, metavar="PATH",
                    help="only read records under this path inside a .zip raw-dir, "
                         "e.g. --prefix physical_sciences/ (the arXiv zip is laid out "
                         "as <domain>/<field>/<paper>.json). Repeatable, so two fields "
                         "can be built into one graph without scanning the rest of the "
                         "corpus.")
    ap.add_argument("--paper-id", action="append", default=None, metavar="ID",
                    help="keep only these OpenAlex ids (e.g. --paper-id W3189559583). "
                         "Repeatable. Useful for extracting one specific subgraph from a "
                         "multi-million-node graph.")
    args = ap.parse_args()
    build_sample(args.raw_dir, args.out, limit=args.limit,
                 coarse_label=not args.full_label, snippet=args.snippet,
                 field=args.field, domain=args.domain,
                 label_source=args.label_source, prefix=args.prefix,
                 paper_ids=args.paper_id)


if __name__ == "__main__":
    main()
