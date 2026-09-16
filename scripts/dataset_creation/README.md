# Knowledge-graph dataset creation

Everything that turns raw paper JSON into a graph lives in this package, kept
separate from the modelling code in [Analysis/](../Analysis/).

```
dataset_creation/
├── schema.py         record parsing + node/edge wiring   (stdlib only)
├── paper_graph.py    build_hetero_graph() -> HeteroData  (Part 2: the reasoning KG)
├── papers.py         PapersDataset        -> one graph per paper (Part 1)
├── build_sample.py   small, embedding-free sample KG you can read
├── visualize.py      renders a *sample* KG to standalone HTML + inline SVG
├── explore.py        renders a *full* KG to a zoomable single-file canvas explorer
├── sample_raw/       3 trimmed real records, enough to run the sample end to end
└── sample/           generated sample output (checked in)
```

---

## Input: what a raw record looks like

The corpus is [`ai4sci-tib/LAION_arxiv-open`](https://huggingface.co/datasets/ai4sci-tib/LAION_arxiv-open):
one `.json` file per paper, where an LLM (`priv-gemini-2.0-flash-lite`) has already
read the full text and emitted a structured summary, joined against OpenAlex
metadata. The builder only ever touches these fields:

| Field | Becomes |
|---|---|
| `summary_title` / `title` + `executive_summary` | the `paper` node's text |
| `field_subfield` (e.g. `"Medicine — Pulmonology"`) | the `field` hub |
| `methodological_details` / `procedures_architectures` | the `method` node |
| `key_results` | the `result` node |
| `claims[]` → `description` / `details` | `claim` nodes |
| `claims[].supporting_evidence` | `evidence` nodes, via `supports` |
| `claims[].implications` | `implication` nodes |
| `openalex_id`, `oa_referenced_works` | `cites` edges |

Three record layouts appear in the wild and all are handled by
`schema._coerce_summary`:

- **Format A/B** — fields nested under `summarization` (which may itself be a
  JSON *string*) → `summary` → `{...}`; claims use the key `details`.
- **Format C** — the summary fields sit flat at the top level; claims use
  `description`. (The shipped `sample_raw/` records are Format C.)
- **Skipped** — `NON_SCIENTIFIC_TEXT` records whose `summary` is `null`, and
  anything missing a field label or a title + executive summary.
  `PARTIAL_SCIENTIFIC_TEXT` records are kept.

`claims` is frequently a *stringified* JSON array; `_get_claims_list` re-parses
it. Same for `oa_referenced_works`.

---

## How the graph is built

`build_hetero_graph(raw_dir, cache_path)` runs two passes.

### Pass 1 — parse and wire (`schema.py`, stdlib only)

`parse_records()` reduces each raw record to a small dict
(`text`, `method`, `result`, `claims[]`, `field`, `refs`, `pid`), then
`build_tables()` flattens those into node text lists and typed edge index lists.

The result is **one big `HeteroData`, not 57k separate graphs** — but each paper
is still its own connected reasoning subgraph, because the edges are typed and
intra-paper. With PyG's `to_hetero`, message passing is already local, so a node
only aggregates from its real neighbours. That gives subgraph semantics without
hierarchical pooling, and keeps a single clean object for the masked-target
reasoning task.

Seven node types:

| Type | Count in the full graph | Cardinality |
|---|---:|---|
| `paper` | 57,903 | 1 per record |
| `field` | 488 | shared hubs |
| `claim` | 251,938 | many per paper |
| `method` | 57,903 | ~1 per paper |
| `result` | 57,903 | ~1 per paper |
| `evidence` | 373,438 † | many per claim |
| `implication` | 251,938 | many per claim |

† That evidence figure predates the schema narrowing to *supporting* evidence
only; it is now smaller. In the 6,191-paper health-sciences slice the graph has
26,857 evidence nodes for 26,858 claims — close to 1:1.

Nine relations — note that two of them are **imposed, not read from the data**:

```
paper  --has_claim-->  claim          paper  --in_field-->  field
paper  --has_method--> method         paper  --cites----->  paper     (DIRECTED, intra-corpus only)
paper  --has_result--> result

method   --produces--> result          ← IMPOSED: every paper's method → its result
result   --grounds---> evidence        ← IMPOSED: the result → every piece of evidence it reports
evidence --supports--> claim           ← claims[].supporting_evidence
claim    --implies---> implication     ← claims[].implications
```

`produces` and `grounds` are the reason a paper is a *chain*

```
paper → method → result → evidence → claim → implication
```

rather than a flat star around the paper node. They encode the assumption that a
paper's method yields its result and its result backs every piece of evidence it
reports — the LLM summary does not state which evidence came from which result,
so the builder wires them completely.

**Evidence is a link in the chain, not a leaf.** It sits *between* the result and
the claim it bears on, so the edge runs `evidence → claim` and the relation is
named `supports` (not `supported_by`, which would read backwards now that the
claim is the destination). Two consequences:

- `contradicting_evidence` is **not read into the graph at all**. Every evidence
  node is supporting evidence, so there is one polarity and one path per claim.
- A claim with no supporting evidence is reachable only through `has_claim` — it
  sits off the chain. `coverage_report()` counts these as
  `claims_without_evidence`, and the builder prints the count; it is 0 or near-0
  on every corpus built so far.

Node rows are assigned in corpus order (files read in sorted order), so the
tables are deterministic given the same inputs.

### Pass 2 — embed (`paper_graph.py`)

Every node's text is encoded with `sentence-transformers/all-MiniLM-L6-v2`
(384-d, batch 256) and becomes that node type's `x`. Nothing is learned here;
this is a frozen featuriser. The whole `HeteroData` plus a `meta` dict of counts
is pickled to `cache_path` (`dataset/papers/processed/hetero_graphA.pt`) and
reused until `REBUILD_HETERO=1`.

Downstream, `train/build_rwse.py` reads this graph back, classifies the
relations into hierarchy / reasoning / other, and computes RWSE positional
encodings over the intra-paper reasoning edges only — which works precisely
because each paper is a disconnected component of that subgraph.

### Things worth knowing

- **`cites` is a directed graph.** An edge `(i, j)` means paper *i* lists paper
  *j* in its own reference list, i.e. **src = citing, dst = cited**. It is the
  only relation whose direction is read from the data rather than imposed, and
  it is stored as a single relation — `Analysis/` training applies
  `T.ToUndirected()`, which synthesises `rev_cites`, so materialising a
  `cited_by` relation here would double those edges. `build_sample.py` writes a
  derived `cited_by` list into `subgraphs.jsonl` for reading convenience; it is
  not a tenth edge type.
- **Citations are sparse, and three things are dropped.** `dangling` —
  references whose OpenAlex id is not in the corpus (the overwhelming majority:
  126,559 of 127,176 in the materials-science slice). `self-citations` — some
  records list their own `openalex_id` in `oa_referenced_works`; an `i→i` edge
  carries no direction, and in the health-sciences slice 91 of 147 citation
  edges were these. `duplicates` — the same reference listed twice in one
  record, which would otherwise weight that pair twice during aggregation. All
  three counts are reported. The builder still warns when this leaves zero
  citation edges.
- **The orientation is checked, not assumed.** `schema.citation_report()`
  compares `oa_year` across each edge and reports how many cite a *newer* paper.
  In the materials-science slice that is 5 of 561 — consistent with preprint and
  OpenAlex date noise, and far from the ~50/50 a reversed edge list would give.
  A silently flipped `src`/`dst` looks exactly like a correct one without this.
- **The field label is coarsened by default.** `"Medicine — Pulmonology"` →
  `"Medicine"`, so all three sample papers share one hub. Pass
  `coarse_label=False` (or `--full-label`) to keep the subfield.
- **`field` nodes have no owner.** They are the only cross-paper structure
  besides `cites`.

---

## The other dataset: `papers.py`

`PapersDataset` is a *different* construction used by Part 1 (Graph-JEPA
pretraining), not by the reasoning KG. It makes **one small graph per paper**:
nodes are the embedded summary sections and claim parts, typed with a richer
14-field vocabulary, and edges are **fully connected** among a paper's nodes.
`y` is the `field_subfield` class id, and classes with fewer than
`min_class_size` papers are dropped. It is kept here because it reads the same
raw records, but it shares no code path with `paper_graph.py`.

---

## Building a sample

No torch, no GPU, no model download — parsing and wiring only:

```bash
python -m dataset_creation.build_sample --snippet 400   # -> dataset_creation/sample/
python -m dataset_creation.visualize                    # -> sample/sample_kg.html
```

### Building one slice, not the whole corpus

The full corpus is 57,903 papers. To prototype on a single slice:

```bash
# one OpenAlex domain — reads the record's top-level `domain` key
python -m dataset_creation.build_sample --raw-dir data/raw \
       --domain "Health Sciences" --out data/kg_health --snippet 400

# one coarse field — matches the label that becomes the graph's `field` hub node
python -m dataset_creation.build_sample --raw-dir data/raw \
       --field Medicine --out data/kg_medicine --snippet 400
```

Both flags are repeatable (`--field Medicine --field Biology`) and are applied
**before** `--limit` counts, so `--field Medicine --limit 500` yields 500
Medicine papers rather than 500 records that happen to contain some. The active
filter is recorded in `stats.json` under `subset`, so two builds from the same
corpus are distinguishable afterwards.

#### Slicing straight out of the 6 GB zip

`--field` has to parse every record before it can reject one. The arXiv zip is
laid out as `<domain>/<field>/<paper>.json`, so `--prefix` rejects out-of-scope
records without ever opening them. It is repeatable, which is how a graph
spanning two fields is built without walking the other 200k records:

```bash
python -m dataset_creation.build_sample \
    --raw-dir dataset_creation/data/LAION_arxiv-open.zip \
    --prefix physical_sciences/materials_science/ \
    --prefix physical_sciences/chemical_engineering/ \
    --label-source field \
    --out dataset_creation/data/kg_materials_chemeng
```

`--label-source field` is **required** on the arXiv corpus. Its `field_subfield`
holds arXiv category codes, not OpenAlex fields — a chemical-engineering record
in this slice is labelled `"Physics — Geophysics"` — so the default would shatter
the graph into dozens of wrong hubs. Reading `field` gives exactly the two hubs
the directory layout promises.

This slice: **6,153 papers → 114,131 nodes, 147,908 edges**, 100% coverage
(every paper has claims, a method and a result), 561 directed citation edges
across 424 citing / 341 cited papers, in ~3 s.

### Exploring a full slice interactively

`visualize.py` hand-authors SVG and tops out around a dozen papers. For a whole
slice, `explore.py` writes a single self-contained HTML file that opens straight
from disk:

```bash
python -m dataset_creation.explore --kg dataset_creation/data/kg_health_sciences
# -> dataset_creation/data/kg_health_sciences/explorer.html
```

Drag to pan, scroll to zoom, hover a node for its text, click to pin it. Search
jumps to a paper by title; the rails filter by field, node type and relation.

Why it is built the way it is — this graph is **not** a hairball. It is ~6.2k
almost-identical little stars (one per paper, mean 18 nodes), a handful of shared
`field` hubs, and a few hundred citation edges. There is no global
structure for a force-directed layout to find, and 113,680 SVG elements would not
render anyway. So:

- **Layout is deterministic and tiled.** Each paper gets a fixed cell laid out
  along its reasoning chain with the same rule as `visualize.instance_svg`; cells
  pack into one block per field. No physics, no layout library, no randomness.
- **Positions are never shipped.** The browser regenerates the layout from a
  compact per-paper structure, so every intra-paper edge is *implied* — ~280 kB of
  structure instead of 147k explicit edge records. Only `cites` is irregular
  enough to need an explicit pair list, and it is drawn with an arrowhead because
  it is the one relation whose direction is real rather than implied by layout.
- **Text is lazy.** All 18.5 MB of node text is gzipped (→ 6.1 MB) into one buffer
  with a `uint32` offset index and decompressed in-browser via
  `DecompressionStream`. Hovering decodes one node's slice, not the whole corpus.
- **Canvas 2D with level-of-detail:** one dot per paper when zoomed out, then all
  nodes, then edges, then wrapped text boxes. The HUD shows the active level.

`field` nodes are drawn as the block headings rather than as nodes, so `in_field`
has no toggle — the grouping *is* the relation. Needs Chrome 80+, Safari 16.4+ or
Firefox 113+ for `DecompressionStream`.

If a filter matches nothing, the parser prints the domains/fields that *were*
present rather than failing silently. Note that `--domain` needs the OpenAlex
`domain` key: the trimmed records in `sample_raw/` don't carry it, so use
`--field` there.

Point it at the real corpus with `--raw-dir data/raw --limit 20`.

Outputs in `sample/`:

| File | Contents |
|---|---|
| `nodes.json` | flat node table — `{id, type, local_idx, owner, text}` |
| `edges.json` | flat edge table — `{src, rel, dst}` |
| `subgraphs.jsonl` | one nested reasoning subgraph per paper, with text |
| `sample_kg.ttl` | the same graph as RDF, matching `Analysis/dataset/papers/export_kg/schema.ttl` |
| `stats.json` | node/edge counts + the parser's coverage report |
| `sample_kg.html` | self-contained visualisation (schema + instance, inline SVG) |

`nodes.json` and `edges.json` are ordinary JSON arrays written **one record per
line**. That is not cosmetic: `json.dump(..., indent=2)` silently disables json's
C encoder, and at slice scale it was 37% of the script's runtime and padded every
record with several lines of indentation. Both files still load with a plain
`json.load`, and stay line-diffable. `stats.json` is small and keeps its
indenting.

The shipped sample is 3 real PubMed papers → **49 nodes, 64 edges**:

```
paper 3   field 1   claim 13   method 3   result 3   evidence 13   implication 13

has_claim 13   has_method 3   has_result 3   in_field 3   cites 0
produces 3   grounds 13   supports 13   implies 13
```

`cites` is 0 because the sample's only intra-corpus reference was a
self-citation, which the directed builder now drops (`56 dangling,
1 self-citation`).

`build_sample.py` calls the same `schema.build_tables()` that
`build_hetero_graph()` does, so the sample's nodes and edges are exactly the ones
the full pipeline would embed.

---

## Running the real build

```python
from dataset_creation.paper_graph import build_hetero_graph

data, meta = build_hetero_graph(
    raw_dir="Analysis/dataset/papers/raw",
    cache_path="Analysis/dataset/papers/processed/hetero_graphA.pt",
    rebuild=True,
)
```

Needs `torch`, `torch_geometric`, `sentence-transformers`. Training scripts under
`Analysis/train/` still import via `core.data_utils.paper_graph`, which is now a
thin shim that re-exports from here.
