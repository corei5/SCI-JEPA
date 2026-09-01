"""
Build a standalone, interactive explorer for a *full* knowledge graph directory
(the output of build_sample.py run over a whole field, e.g. data/kg_health_sciences).

Why this is not just `visualize.py` with more data
--------------------------------------------------
`visualize.py` hand-authors inline SVG. That works for three papers and dies at
six thousand: 113k <rect> elements is far past what any browser will lay out, and
a force-directed layout would be meaningless anyway — this graph is not a hairball.
It is ~6.2k almost-identical little stars (one per paper, mean 18 nodes), five
shared `field` hubs, and a handful of citation edges. There is no global structure
for a physics simulation to discover.

So this module takes the opposite approach:

  * **Deterministic tiled layout.** Every paper gets a fixed cell, laid out along
    its reasoning chain (method -> result -> evidence -> claim -> implication) using
    the *same* algorithm as visualize.instance_svg. Cells are packed into one
    block per field. No physics, no layout library, no randomness.
  * **Nothing is positioned in Python.** The browser regenerates the layout from a
    compact structural description, so we ship ~6.2k small arrays instead of
    147k explicit edge records.
  * **Canvas 2D with level-of-detail**, not SVG. Zoomed out you get one dot per
    paper; zoom in and nodes, then edges, then wrapped text boxes appear.
  * **Text is decoded lazily.** All node text is gzipped into one buffer with a
    uint32 offset index. Hovering a node decodes just that node's slice.

The result is one self-contained .html that opens straight from disk.

Usage
-----
    python -m dataset_creation.explore --kg dataset_creation/data/kg_health_sciences
    # -> dataset_creation/data/kg_health_sciences/explorer.html
"""

import argparse
import base64
import gzip
import json
import os
import struct

from .schema import NODE_TYPES
from .visualize import PALETTE_DARK, PALETTE_LIGHT, REL_STYLE, TYPE_COLOR, esc

HERE = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
#  Payload 1 — the text table
# ===========================================================================
def build_text_table(nodes):
    """
    Pack every node's text into one gzipped blob with a random-access index.

    Layout of the *uncompressed* buffer:

        uint32   n                 number of nodes
        uint32   off[0 .. n]       byte offsets into `blob`, n+1 of them
        bytes    blob              all texts concatenated, utf-8, no separators

    Node i's text is blob[off[i] : off[i+1]]. The browser gunzips this once and
    then TextDecoder-slices individual nodes on demand, which is what keeps
    hovering cheap: we never materialise 113k JavaScript strings.
    """
    encoded = [n["text"].encode("utf-8") for n in nodes]
    offsets, cur = [0], 0
    for b in encoded:
        cur += len(b)
        offsets.append(cur)

    header = struct.pack("<I", len(encoded)) + struct.pack(f"<{len(offsets)}I", *offsets)
    raw = header + b"".join(encoded)
    return gzip.compress(raw, 9), len(raw)


# ===========================================================================
#  Payload 2 — the structural description
# ===========================================================================
def build_structure(nodes, edges):
    """
    Reduce the edge list to one nested array per paper.

    Every intra-paper relation is *implied* by a paper's shape, so we never send
    the edge list itself:

        has_method    paper -> m
        has_result    paper -> r
        produces      m     -> r
        has_claim     paper -> each claim
        grounds       r     -> each id in every claim's `s` list
        supports      each id in a claim's `s` list -> that claim
        implies       claim -> each id in its `p` list
        in_field      paper -> f

    `s` is the claim's supporting evidence. It is stored under the claim purely
    because that is the compact way to write it down — the edges themselves run
    result -> evidence -> claim, i.e. through the evidence, not around it.

    Only `cites` is genuinely irregular, so it stays an explicit pair list.

    Indices stored here are *local* (rank within a node type). The browser adds
    the per-type base offset to get an index into the text table. Local indices
    are small and highly repetitive, which is why this gzips down so well.
    """
    by_id = {n["id"]: n for n in nodes}
    local = {n["id"]: n["local_idx"] for n in nodes}

    # ---- bucket the edges we need, keyed by source id
    method_of, result_of, field_of = {}, {}, {}
    claims_of = {}
    leaves_of = {}          # claim id -> {"s": [evidence...], "p": [implication...]}
    cites = []

    def leaves(cid):
        return leaves_of.setdefault(cid, {"s": [], "p": []})

    for e in edges:
        rel, src, dst = e["rel"], e["src"], e["dst"]
        if rel == "has_method":
            method_of[src] = dst
        elif rel == "has_result":
            result_of[src] = dst
        elif rel == "in_field":
            field_of[src] = dst
        elif rel == "has_claim":
            claims_of.setdefault(src, []).append(dst)
        elif rel == "supports":
            # evidence -> claim, so the CLAIM is the destination here
            leaves(dst)["s"].append(src)
        elif rel == "implies":
            leaves(src)["p"].append(dst)
        elif rel == "cites":
            cites.append((local[src], local[dst]))
        # `grounds` (result -> evidence) is not stored: it is implied by the
        # paper's result plus the evidence already listed under each claim.

    papers = [n["id"] for n in nodes if n["type"] == "paper"]
    out = []
    for p in papers:
        claim_rows = []
        for c in claims_of.get(p, []):
            lv = leaves_of.get(c, {"s": [], "p": []})
            claim_rows.append([
                local[c],
                [local[x] for x in lv["s"]],
                [local[x] for x in lv["p"]],
            ])
        out.append([
            local[field_of[p]] if p in field_of else -1,
            local[method_of[p]] if p in method_of else -1,
            local[result_of[p]] if p in result_of else -1,
            claim_rows,
        ])

    # ---- per-type base offsets into the text table (nodes.json order)
    bases, seen = {}, set()
    for i, n in enumerate(nodes):
        if n["type"] not in seen:
            seen.add(n["type"])
            bases[n["type"]] = i

    fields = [n["text"] for n in nodes if n["type"] == "field"]

    return {
        "bases": bases,
        "fields": fields,
        "papers": out,
        "cites": cites,
        "selfCites": sum(1 for a, b in cites if a == b),
    }


# ===========================================================================
#  Page assembly
# ===========================================================================
def _vars(p):
    return "\n".join(f"  {k}: {v};" for k, v in p.items())


def _b64(raw_bytes):
    return base64.b64encode(raw_bytes).decode("ascii")


def build_page(kg_dir, out_path, title=None):
    nodes = json.load(open(os.path.join(kg_dir, "nodes.json"), encoding="utf-8"))
    edges = json.load(open(os.path.join(kg_dir, "edges.json"), encoding="utf-8"))
    stats = json.load(open(os.path.join(kg_dir, "stats.json"), encoding="utf-8"))

    title = title or (os.path.basename(kg_dir.rstrip("/")).replace("kg_", "")
                      .replace("_", " ").title() + " Knowledge Graph")

    text_gz, text_raw_len = build_text_table(nodes)
    struct_obj = build_structure(nodes, edges)
    struct_gz = gzip.compress(json.dumps(struct_obj, separators=(",", ":")).encode("utf-8"), 9)

    meta = {
        "title": title,
        "nodeTypes": list(NODE_TYPES),
        "relStyle": {r: [v, dashed, gloss] for r, (v, dashed, gloss) in REL_STYLE.items()},
        "typeColor": TYPE_COLOR,
        "stats": stats,
    }

    html_doc = (PAGE
                .replace("__TITLE__", esc(title))
                .replace("__CSS_LIGHT__", _vars(PALETTE_LIGHT))
                .replace("__CSS_DARK__", _vars(PALETTE_DARK))
                .replace("__META__", json.dumps(meta, separators=(",", ":")))
                .replace("__STRUCT_B64__", _b64(struct_gz))
                .replace("__TEXT_B64__", _b64(text_gz)))

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    mb = os.path.getsize(out_path) / 1e6
    print(f"[explore] {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    print(f"[explore] text   {text_raw_len / 1e6:6.1f} MB -> {len(text_gz) / 1e6:.1f} MB gzip")
    print(f"[explore] struct {len(struct_gz) / 1e3:6.1f} kB gzip "
          f"({len(struct_obj['papers'])} papers, {len(struct_obj['cites'])} cites, "
          f"{struct_obj['selfCites']} of them self-loops)")
    print(f"[explore] wrote {out_path}  ({mb:.1f} MB)")
    return out_path


# ===========================================================================
#  The page template.  __PLACEHOLDERS__ are substituted above; the file is
#  plain text here so JS/CSS braces need no escaping.
# ===========================================================================
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
__CSS_LIGHT__
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
__CSS_DARK__
  }
}
:root[data-theme="dark"] {
__CSS_DARK__
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; overflow: hidden; }
body {
  background: var(--bg); color: var(--fg);
  font: 14px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
canvas { display: block; position: absolute; inset: 0; cursor: grab; }
canvas.dragging { cursor: grabbing; }

.panel {
  position: absolute; background: var(--panel); border: 1px solid var(--rule);
  border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,.13);
  backdrop-filter: blur(8px);
}
h1 { font-size: 15px; margin: 0 0 2px; letter-spacing: -0.01em; }
h3 { font-size: 10.5px; text-transform: uppercase; letter-spacing: .07em;
     color: var(--muted); margin: 14px 0 6px; font-weight: 600; }
code { font: 11.5px ui-monospace, SFMono-Regular, Menlo, monospace; }

/* ---------- left control rail ---------- */
#rail { top: 14px; left: 14px; width: 268px; max-height: calc(100% - 28px);
        overflow-y: auto; padding: 14px 15px 16px; }
#rail .sub { color: var(--muted); font-size: 11.5px; margin: 0 0 2px; }
#search { width: 100%; padding: 6px 9px; border-radius: 7px; font: inherit;
          font-size: 12.5px; border: 1px solid var(--rule);
          background: var(--bg); color: var(--fg); }
#search:focus { outline: 2px solid var(--c-paper); outline-offset: -1px; }
#results { list-style: none; margin: 6px 0 0; padding: 0; max-height: 210px;
           overflow-y: auto; font-size: 12px; }
#results li { padding: 5px 7px; border-radius: 6px; cursor: pointer;
              border-left: 3px solid transparent; }
#results li:hover { background: var(--bg); border-left-color: var(--c-paper); }
#results .rf { color: var(--muted); font-size: 10.5px; }

ul.toggles { list-style: none; margin: 0; padding: 0; }
ul.toggles li { display: flex; align-items: center; gap: 7px; padding: 2.5px 0;
                font-size: 12.5px; cursor: pointer; user-select: none; }
ul.toggles input { margin: 0; accent-color: var(--c-paper); flex: none; }
.dot { width: 11px; height: 11px; border-radius: 3px; border: 1.5px solid; flex: none; }
.bar { width: 22px; height: 0; flex: none; }
.count { margin-left: auto; color: var(--muted); font-size: 10.5px;
         font-variant-numeric: tabular-nums; }
.gloss { color: var(--muted); font-size: 11px; }

/* ---------- detail panel ---------- */
#detail { top: 14px; right: 14px; width: 336px; max-height: calc(100% - 190px);
          overflow-y: auto; padding: 14px 15px 16px; display: none; }
#detail.open { display: block; }
#detail .badge { display: inline-block; padding: 1px 8px; border-radius: 999px;
                 font-size: 10.5px; font-weight: 600; border: 1.5px solid;
                 letter-spacing: .03em; }
#detail .body { margin: 9px 0 0; font-size: 13px; white-space: pre-wrap;
                word-break: break-word; }
#detail .meta { color: var(--muted); font-size: 11.5px; margin-top: 10px;
                border-top: 1px solid var(--rule); padding-top: 9px; }
#detail .meta div { display: flex; gap: 8px; padding: 1px 0; }
#detail .meta span:first-child { color: var(--muted); min-width: 74px; }
#close { position: absolute; top: 10px; right: 11px; border: 0; background: none;
         color: var(--muted); font-size: 17px; cursor: pointer; line-height: 1;
         padding: 2px 4px; }
#close:hover { color: var(--fg); }

/* ---------- tooltip ---------- */
#tip { position: absolute; pointer-events: none; max-width: 380px; padding: 8px 11px;
       font-size: 12px; line-height: 1.45; border-radius: 8px; z-index: 40;
       background: var(--panel); border: 1px solid var(--rule);
       box-shadow: 0 6px 20px rgba(0,0,0,.22); display: none; }
#tip .badge { font-size: 10px; font-weight: 700; text-transform: uppercase;
              letter-spacing: .06em; }
#tip .txt { margin-top: 3px; color: var(--fg); }

/* ---------- bottom-right cluster ---------- */
#minimap { bottom: 14px; right: 14px; padding: 7px; }
#minimap canvas { position: static; border-radius: 5px; cursor: crosshair; }
#hud { bottom: 14px; right: 260px; padding: 7px 9px; display: flex; gap: 5px;
       align-items: center; font-size: 11.5px; }
#hud button { border: 1px solid var(--rule); background: var(--bg); color: var(--fg);
              border-radius: 6px; width: 26px; height: 26px; cursor: pointer;
              font-size: 14px; line-height: 1; padding: 0; }
#hud button:hover { border-color: var(--c-paper); color: var(--c-paper); }
#hud button.wide { width: auto; padding: 0 9px; font-size: 11.5px; }
#zoomlvl { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 66px;
           text-align: right; }

#loading { position: absolute; inset: 0; display: grid; place-items: center;
           background: var(--bg); z-index: 90; font-size: 13px; color: var(--muted); }
#hint { position: absolute; bottom: 16px; left: 14px; color: var(--muted);
        font-size: 11.5px; pointer-events: none; }
</style>
</head>
<body>

<canvas id="cv"></canvas>

<div id="rail" class="panel">
  <h1>__TITLE__</h1>
  <p class="sub" id="summary"></p>

  <h3>Find a paper</h3>
  <input id="search" type="search" placeholder="title keywords…" autocomplete="off">
  <ul id="results"></ul>

  <h3>Fields</h3>
  <ul class="toggles" id="fieldToggles"></ul>

  <h3>Node types</h3>
  <ul class="toggles" id="typeToggles"></ul>

  <h3>Relations</h3>
  <ul class="toggles" id="relToggles"></ul>
</div>

<div id="detail" class="panel">
  <button id="close" title="Close">&times;</button>
  <span class="badge" id="dBadge"></span>
  <p class="body" id="dBody"></p>
  <div class="meta" id="dMeta"></div>
</div>

<div id="tip"><span class="badge"></span><div class="txt"></div></div>

<div id="hud" class="panel">
  <button id="zOut" title="Zoom out (-)">&minus;</button>
  <button id="zIn" title="Zoom in (+)">+</button>
  <button id="fit" class="wide" title="Fit whole graph (0)">Fit</button>
  <span id="zoomlvl"></span>
</div>

<div id="minimap" class="panel"><canvas id="mm" width="216" height="132"></canvas></div>

<div id="hint">drag to pan · scroll to zoom · hover for text · click to pin</div>
<div id="loading">decompressing graph…</div>

<!-- The payload must be parsed before the app script: separate <script> tags run
     in order and `const` does not hoist across them. -->
<script id="payload">
const STRUCT_B64 = "__STRUCT_B64__";
const TEXT_B64 = "__TEXT_B64__";
</script>

<script>
"use strict";
const META = __META__;

/* =====================================================================
   0.  Payload decoding
   ===================================================================== */
function b64ToBytes(s) {
  const bin = atob(s), n = bin.length, out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
  return out;
}
async function gunzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/* Random-access view over the concatenated node texts.  We hold the raw utf-8
   bytes and only decode the slice a node actually needs, so hovering never
   costs more than a few hundred bytes of work. */
class TextTable {
  constructor(buf) {
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    this.n = dv.getUint32(0, true);
    this.off = new Uint32Array(buf.buffer, buf.byteOffset + 4, this.n + 1);
    this.blob = buf.subarray(4 + (this.n + 1) * 4);
    this.dec = new TextDecoder("utf-8");
    this.cache = new Map();
  }
  get(i) {
    if (i < 0 || i >= this.n) return "";
    let v = this.cache.get(i);
    if (v === undefined) {
      v = this.dec.decode(this.blob.subarray(this.off[i], this.off[i + 1]));
      if (this.cache.size > 4000) this.cache.clear();
      this.cache.set(i, v);
    }
    return v;
  }
}

/* =====================================================================
   1.  World layout
   ---------------------------------------------------------------------
   Deterministic and reproduced entirely in the browser.  Each paper owns a
   fixed CELL_W x CELL_H cell; inside it the reasoning chain runs left to
   right, and leaves stack vertically with parents centred on their children
   (the same rule visualize.instance_svg uses).  Cells are packed into a
   roughly square block per field, and the blocks sit side by side.
   ===================================================================== */
const CELL_W = 2000, CELL_H = 1040;
const PITCH_X = CELL_W + 250, PITCH_Y = CELL_H + 240;
const BLOCK_GAP = PITCH_X * 2.2, BLOCK_HEAD = PITCH_Y * 0.75;
const LEAF_PITCH = 46, CLAIM_GAP = 14;

/* column x offset and box width, in world units, per node type.
   A node's stored `x` is always the LEFT edge of its box.  Dots are drawn at the
   box centre and edges run right-edge -> left-edge, so the two level-of-detail
   modes line up instead of drifting apart as you zoom.
   Column order is the reasoning chain: evidence sits between result and claim,
   and implication is the only leaf column. */
const COL = {
  paper:       [  40, 250], method:      [ 330, 250], result:      [ 620, 250],
  evidence:    [ 910, 300], claim:       [1260, 290], implication: [1600, 330],
  field:       [   0, 520],
};
const cx_of = (n) => n.x + COL[n.t][1] / 2;      // centre  (dots, hit test)
const rx_of = (n) => n.x + COL[n.t][1];          // right   (edge source)
const lx_of = (n) => n.x;                        // left    (edge target)

const R_NODE = { paper: 15, method: 10, result: 10, claim: 11,
                 evidence: 8, implication: 8, field: 30 };

/* Distinct hues for the field blocks, used only where colour has to encode
   *which field* rather than which node type (the zoomed-out dots and minimap). */
const FIELD_HUE = [212, 152, 28, 330, 265, 190, 95, 15];
const fieldColor = (bi, a) => `hsl(${FIELD_HUE[bi % FIELD_HUE.length]} 62% 52% / ${a})`;

/* Level-of-detail thresholds, measured in on-screen pixels per cell.  Below
   L_NODES a paper is a single dot and we never even build its layout. */
const L_NODES = 46, L_EDGES = 300, L_BOXES = 950;

let TT = null;            // TextTable
let S = null;             // structure payload
let BASE = null;          // node-type -> text-table base index
let BLOCKS = [];          // one per field
let PAPER_BLOCK = null;   // paper idx -> block idx
let PAPER_CELL = null;    // paper idx -> [col, row] within its block
let TITLES = null;        // paper idx -> short title (for search)
const LAYOUT = new Map(); // paper idx -> {nodes:[{t,x,y,ti,id}], edges:[[a,b,rel]]}

function buildBlocks() {
  const byField = new Map();
  S.papers.forEach((p, i) => {
    const f = p[0];
    if (!byField.has(f)) byField.set(f, []);
    byField.get(f).push(i);
  });

  /* biggest field first so the layout reads left to right by size */
  const order = [...byField.entries()].sort((a, b) => b[1].length - a[1].length);

  PAPER_BLOCK = new Int32Array(S.papers.length);
  PAPER_CELL = new Int32Array(S.papers.length * 2);

  let x = 0;
  BLOCKS = order.map(([fieldIdx, members], bi) => {
    const cols = Math.max(1, Math.ceil(Math.sqrt(members.length * PITCH_Y / PITCH_X)));
    const rows = Math.ceil(members.length / cols);
    members.forEach((pi, k) => {
      PAPER_BLOCK[pi] = bi;
      PAPER_CELL[pi * 2] = k % cols;
      PAPER_CELL[pi * 2 + 1] = (k / cols) | 0;
    });
    const b = {
      fieldIdx, members, cols, rows, x, y: 0,
      w: cols * PITCH_X, h: rows * PITCH_Y,
      name: S.fields[fieldIdx],
    };
    x += b.w + BLOCK_GAP;
    return b;
  });
}

function cellOrigin(pi) {
  const b = BLOCKS[PAPER_BLOCK[pi]];
  return [b.x + PAPER_CELL[pi * 2] * PITCH_X, b.y + PAPER_CELL[pi * 2 + 1] * PITCH_Y];
}

/* Build (and memoise) the node/edge geometry for one paper. */
function layoutOf(pi) {
  let L = LAYOUT.get(pi);
  if (L) return L;

  const [fieldIdx, mLocal, rLocal, claims] = S.papers[pi];   /* claim = [local, evidence[], implication[]] */
  const nodes = [], edges = [];
  const push = (t, base, local, x, y) =>
    nodes.push({ t, x, y, ti: base + local, local }) - 1;

  /* vertical pass: rows first, then the claim centred on its own rows.
     A claim's evidence (left of it) and implications (right of it) SHARE the
     rows it reserves — they are in different columns, so the block is only as
     tall as the taller of the two stacks. */
  let cursor = 0;
  const claimY = [];
  const leafRows = [];
  for (const [cLocal, sup, imp] of claims) {
    const rows = Math.max(sup.length, imp.length, 1);
    const ys = [];
    for (let k = 0; k < rows; k++) { ys.push(cursor); cursor += LEAF_PITCH; }
    claimY.push((ys[0] + ys[rows - 1]) / 2);
    leafRows.push([sup, imp, ys]);
    cursor += CLAIM_GAP;
  }
  const contentH = Math.max(cursor, LEAF_PITCH);
  const pY = claimY.length ? claimY.reduce((a, b) => a + b, 0) / claimY.length : contentH / 2;

  /* squeeze into the cell only if this paper actually overflows it */
  const margin = 60;
  const k = Math.min(1, (CELL_H - 2 * margin) / contentH);
  const shift = (CELL_H - contentH * k) / 2;
  const Y = (v) => shift + v * k;

  const iPaper = push("paper", BASE.paper, pi, COL.paper[0], Y(pY));
  const iMethod = mLocal >= 0 ? push("method", BASE.method, mLocal, COL.method[0], Y(pY) - 95 * k) : -1;
  const iResult = rLocal >= 0 ? push("result", BASE.result, rLocal, COL.result[0], Y(pY)) : -1;

  if (iMethod >= 0) edges.push([iPaper, iMethod, "has_method"]);
  if (iResult >= 0) edges.push([iPaper, iResult, "has_result"]);
  if (iMethod >= 0 && iResult >= 0) edges.push([iMethod, iResult, "produces"]);

  claims.forEach(([cLocal], ci) => {
    const iClaim = push("claim", BASE.claim, cLocal, COL.claim[0], Y(claimY[ci]));
    edges.push([iPaper, iClaim, "has_claim"]);
    const [sup, imp, ys] = leafRows[ci];
    /* the chain reaches the claim THROUGH its evidence: result -> evidence -> claim */
    sup.forEach((local, li) => {
      const iEv = push("evidence", BASE.evidence, local, COL.evidence[0], Y(ys[li]));
      if (iResult >= 0) edges.push([iResult, iEv, "grounds"]);
      edges.push([iEv, iClaim, "supports"]);
    });
    imp.forEach((local, li) => {
      const iImp = push("implication", BASE.implication, local, COL.implication[0], Y(ys[li]));
      edges.push([iClaim, iImp, "implies"]);
    });
  });

  L = { nodes, edges, fieldIdx };
  if (LAYOUT.size > 4000) LAYOUT.clear();
  LAYOUT.set(pi, L);
  return L;
}

/* =====================================================================
   2.  Camera
   ===================================================================== */
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
let W = 0, H = 0, DPR = 1;
const cam = { x: 0, y: 0, s: 1 };   // world point at screen centre, and scale

const toScreenX = (wx) => (wx - cam.x) * cam.s + W / 2;
const toScreenY = (wy) => (wy - cam.y) * cam.s + H / 2;
const toWorldX = (sx) => (sx - W / 2) / cam.s + cam.x;
const toWorldY = (sy) => (sy - H / 2) / cam.s + cam.y;

function worldBounds() {
  const last = BLOCKS[BLOCKS.length - 1];
  return {
    x0: -BLOCK_GAP * 0.3, y0: -BLOCK_HEAD * 1.4,
    x1: last.x + last.w + BLOCK_GAP * 0.3,
    y1: Math.max(...BLOCKS.map((b) => b.y + b.h)) + PITCH_Y * 0.3,
  };
}
function fit(pad = 0.94) {
  const b = worldBounds();
  cam.s = Math.min(W / (b.x1 - b.x0), H / (b.y1 - b.y0)) * pad;
  cam.x = (b.x0 + b.x1) / 2;
  cam.y = (b.y0 + b.y1) / 2;
  draw();
}
function flyToPaper(pi, scale) {
  const [ox, oy] = cellOrigin(pi);
  cam.x = ox + CELL_W / 2; cam.y = oy + CELL_H / 2;
  cam.s = scale !== undefined ? scale : L_BOXES / CELL_W * 1.15;
  draw();
}

/* =====================================================================
   3.  Theme colours — resolved from CSS variables so light/dark just work
   ===================================================================== */
let CSS = {};
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  CSS = {};
  for (const t of META.nodeTypes) {
    CSS[t] = cs.getPropertyValue(META.typeColor[t]).trim();
    CSS[t + "-bg"] = cs.getPropertyValue(META.typeColor[t] + "-bg").trim();
  }
  for (const r in META.relStyle) CSS[r] = cs.getPropertyValue(META.relStyle[r][0]).trim();
  for (const k of ["--bg", "--fg", "--muted", "--rule", "--panel"])
    CSS[k] = cs.getPropertyValue(k).trim();
}

/* =====================================================================
   4.  State + filters
   ===================================================================== */
const showType = {}, showRel = {}, showField = {};
let hover = null;      // {pi, ni} or {pi:-1, fieldIdx} for a field hub
let pinned = null;

/* =====================================================================
   5.  Drawing
   ===================================================================== */
function visiblePapers() {
  const x0 = toWorldX(0), x1 = toWorldX(W), y0 = toWorldY(0), y1 = toWorldY(H);
  const out = [];
  for (let bi = 0; bi < BLOCKS.length; bi++) {
    const b = BLOCKS[bi];
    if (!showField[b.fieldIdx]) continue;
    if (b.x > x1 || b.x + b.w < x0 || b.y > y1 || b.y + b.h < y0) continue;
    const c0 = Math.max(0, Math.floor((x0 - b.x) / PITCH_X));
    const c1 = Math.min(b.cols - 1, Math.floor((x1 - b.x) / PITCH_X));
    const r0 = Math.max(0, Math.floor((y0 - b.y) / PITCH_Y));
    const r1 = Math.min(b.rows - 1, Math.floor((y1 - b.y) / PITCH_Y));
    for (let r = r0; r <= r1; r++)
      for (let c = c0; c <= c1; c++) {
        const k = r * b.cols + c;
        if (k < b.members.length) out.push(b.members[k]);
      }
  }
  return out;
}

const wrapCache = new Map();
function wrapText(ti, text, widthPx, maxLines, font) {
  const key = ti + "|" + (widthPx | 0) + "|" + maxLines;
  let v = wrapCache.get(key);
  if (v) return v;
  ctx.font = font;
  const words = text.split(/\s+/);
  const lines = [];
  let cur = "";
  for (const w of words) {
    const cand = cur ? cur + " " + w : w;
    if (ctx.measureText(cand).width <= widthPx) { cur = cand; continue; }
    if (cur) lines.push(cur);
    cur = w;
    if (lines.length >= maxLines) break;
  }
  if (cur && lines.length < maxLines) lines.push(cur);
  if (lines.length >= maxLines) {
    let last = lines[maxLines - 1];
    while (last.length > 1 && ctx.measureText(last + "…").width > widthPx) last = last.slice(0, -1);
    lines[maxLines - 1] = last + "…";
  }
  if (wrapCache.size > 3000) wrapCache.clear();
  wrapCache.set(key, lines);
  return lines;
}

function draw() {
  if (!S) return;
  const cellPx = CELL_W * cam.s;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.fillStyle = CSS["--bg"];
  ctx.fillRect(0, 0, W, H);

  const papers = visiblePapers();

  /* ---- field block frames + labels ---- */
  for (const b of BLOCKS) {
    if (!showField[b.fieldIdx]) continue;
    const sx = toScreenX(b.x), sy = toScreenY(b.y);
    const sw = b.w * cam.s, sh = b.h * cam.s;
    if (sx > W || sy > H || sx + sw < 0 || sy + sh < 0) continue;
    ctx.strokeStyle = CSS["--rule"];
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 5]);
    ctx.strokeRect(sx - 10, sy - 10, sw + 20, sh + 20);
    ctx.setLineDash([]);
    const fs = Math.max(12, Math.min(34, sw * 0.035));
    ctx.font = `600 ${fs}px ui-sans-serif, -apple-system, sans-serif`;
    ctx.fillStyle = CSS["--muted"];
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillText(`${b.name}  ·  ${b.members.length} papers`,
                 Math.max(8, sx - 10), Math.max(fs + 6, sy - 20));
  }

  /* ---- level 0: one dot per paper, coloured by field ---- */
  if (cellPx < L_NODES) {
    const r = Math.max(1, Math.min(5, cellPx * 0.16));
    let curBlock = -1;
    for (const pi of papers) {
      const bi = PAPER_BLOCK[pi];
      if (bi !== curBlock) { curBlock = bi; ctx.fillStyle = fieldColor(bi, 0.85); }
      const [ox, oy] = cellOrigin(pi);
      const x = toScreenX(ox + CELL_W / 2), y = toScreenY(oy + CELL_H / 2);
      ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }
    drawCites(cellPx);
    drawHover(cellPx);
    drawMinimap();
    updateHud(papers.length);
    return;
  }

  /* ---- levels 1-3 ---- */
  const drawEdges = cellPx >= L_EDGES;
  const drawBoxes = cellPx >= L_BOXES;

  if (drawEdges) {
    ctx.lineWidth = Math.max(0.6, Math.min(2, cellPx / 900));
    ctx.globalAlpha = 0.7;
    for (const pi of papers) {
      const L = layoutOf(pi);
      const [ox, oy] = cellOrigin(pi);
      for (const [a, b, rel] of L.edges) {
        if (!showRel[rel]) continue;
        const na = L.nodes[a], nb = L.nodes[b];
        if (!showType[na.t] || !showType[nb.t]) continue;
        /* in dot mode anchor on the dots themselves, in box mode on the box edges */
        const x1 = toScreenX(ox + (drawBoxes ? rx_of(na) : cx_of(na)));
        const x2 = toScreenX(ox + (drawBoxes ? lx_of(nb) : cx_of(nb)));
        const y1 = toScreenY(oy + na.y), y2 = toScreenY(oy + nb.y);
        ctx.strokeStyle = CSS[rel];
        ctx.setLineDash(META.relStyle[rel][1] ? [5, 4] : []);
        ctx.beginPath();
        const dx = Math.max(20, Math.abs(x2 - x1) * 0.45);
        ctx.moveTo(x1, y1);
        ctx.bezierCurveTo(x1 + dx, y1, x2 - dx, y2, x2, y2);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  for (const pi of papers) {
    const L = layoutOf(pi);
    const [ox, oy] = cellOrigin(pi);
    for (const n of L.nodes) {
      if (!showType[n.t]) continue;
      const cy = toScreenY(oy + n.y);
      const cx = toScreenX(ox + (drawBoxes ? lx_of(n) : cx_of(n)));
      if (cx < -400 || cx > W + 400 || cy < -100 || cy > H + 100) continue;
      if (drawBoxes) {
        const w = COL[n.t][1] * cam.s;
        const text = TT.get(n.ti);
        const fs = Math.max(7, 11 * cam.s * (CELL_W / 1650));
        const font = `${fs}px ui-sans-serif, -apple-system, sans-serif`;
        const lines = wrapText(n.ti, text, w - 14 * cam.s, n.t === "paper" ? 4 : 3, font);
        const h = lines.length * fs * 1.28 + 12 * cam.s;
        ctx.fillStyle = CSS[n.t + "-bg"];
        ctx.strokeStyle = CSS[n.t];
        ctx.lineWidth = 1.3;
        roundRect(cx, cy - h / 2, w, h, 6 * cam.s);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = CSS["--fg"];
        ctx.font = font;
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        lines.forEach((ln, i) => {
          ctx.fillText(ln, cx + 7 * cam.s,
                       cy - h / 2 + 6 * cam.s + (i + 0.5) * fs * 1.28);
        });
      } else {
        const r = Math.max(1, R_NODE[n.t] * cam.s);
        ctx.fillStyle = CSS[n.t];
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  drawCites(cellPx);
  drawHover(cellPx);
  drawMinimap();
  updateHud(papers.length);
}

function roundRect(x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* The only real cross-paper structure, so it gets drawn on top of everything
   and is never culled — otherwise you would have to already know where to look. */
function drawCites(cellPx) {
  if (!showRel.cites) return;
  ctx.strokeStyle = CSS.cites;
  ctx.lineWidth = Math.max(1, Math.min(2.5, cellPx / 500));
  ctx.globalAlpha = 0.9;
  for (const [a, b] of S.cites) {
    if (!showField[S.papers[a][0]] || !showField[S.papers[b][0]]) continue;
    const [ax, ay] = cellOrigin(a), [bx, by] = cellOrigin(b);
    const x1 = toScreenX(ax + CELL_W / 2), y1 = toScreenY(ay + CELL_H / 2);
    if (a === b) {                                   // self-citation
      const r = Math.max(3, CELL_W * 0.22 * cam.s);
      ctx.beginPath();
      ctx.arc(x1, y1 - r * 0.8, r, 0.15 * Math.PI, 0.85 * Math.PI, true);
      ctx.stroke();
      continue;
    }
    const x2 = toScreenX(bx + CELL_W / 2), y2 = toScreenY(by + CELL_H / 2);
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 - Math.hypot(x2 - x1, y2 - y1) * 0.18;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(mx, my, x2, y2);
    ctx.stroke();
    /* `cites` is directed (citing -> cited) and it is the only relation drawn as
       a bare curve, so without a head the two directions are indistinguishable.
       The tangent of a quadratic at t=1 is (end - control). */
    drawArrowHead(x1, y1, mx, my, x2, y2);
  }
  ctx.globalAlpha = 1;
}

function drawArrowHead(x1, y1, mx, my, x2, y2) {
  const tx = x2 - mx, ty = y2 - my;
  const len = Math.hypot(tx, ty);
  if (len < 1) return;
  const ux = tx / len, uy = ty / len;
  const h = Math.max(4, Math.min(11, ctx.lineWidth * 5));
  const w = h * 0.45;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - ux * h - uy * w, y2 - uy * h + ux * w);
  ctx.lineTo(x2 - ux * h + uy * w, y2 - uy * h - ux * w);
  ctx.closePath();
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();
}

function drawHover(cellPx) {
  const target = pinned || hover;
  if (!target || target.ni === undefined) return;
  const L = layoutOf(target.pi);
  const n = L.nodes[target.ni];
  if (!n) return;
  const [ox, oy] = cellOrigin(target.pi);
  const boxes = cellPx >= L_BOXES;
  const cy = toScreenY(oy + n.y);
  const cx = toScreenX(ox + (boxes ? lx_of(n) : cx_of(n)));

  /* incident edges of the focused node, drawn bright so the local
     reasoning chain pops out of the surrounding grid */
  ctx.lineWidth = 2.2; ctx.globalAlpha = 1;
  ctx.setLineDash([]);
  for (const [a, b, rel] of L.edges) {
    if (a !== target.ni && b !== target.ni) continue;
    const na = L.nodes[a], nb = L.nodes[b];
    const x1 = toScreenX(ox + (boxes ? rx_of(na) : cx_of(na)));
    const x2 = toScreenX(ox + (boxes ? lx_of(nb) : cx_of(nb)));
    const y1 = toScreenY(oy + na.y), y2 = toScreenY(oy + nb.y);
    ctx.strokeStyle = CSS[rel];
    ctx.beginPath();
    const dx = Math.max(20, Math.abs(x2 - x1) * 0.45);
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(x1 + dx, y1, x2 - dx, y2, x2, y2);
    ctx.stroke();
  }

  ctx.strokeStyle = CSS[n.t];
  ctx.lineWidth = 2.5;
  if (boxes) {
    const w = COL[n.t][1] * cam.s;
    roundRect(cx - 3, cy - 34 * cam.s, w + 6, 68 * cam.s, 7 * cam.s);
    ctx.stroke();
  } else {
    const r = Math.max(6, R_NODE[n.t] * cam.s * 1.4);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
  }
}

/* =====================================================================
   6.  Minimap
   ===================================================================== */
const mm = document.getElementById("mm"), mmc = mm.getContext("2d");
function drawMinimap() {
  const b = worldBounds();
  const sw = mm.width, sh = mm.height;
  const k = Math.min(sw / (b.x1 - b.x0), sh / (b.y1 - b.y0)) * 0.92;
  const offX = (sw - (b.x1 - b.x0) * k) / 2, offY = (sh - (b.y1 - b.y0) * k) / 2;
  mm.__k = k; mm.__off = [offX, offY]; mm.__b = b;

  mmc.fillStyle = CSS["--bg"];
  mmc.fillRect(0, 0, sw, sh);
  BLOCKS.forEach((bl, bi) => {
    const on = showField[bl.fieldIdx];
    mmc.fillStyle = on ? fieldColor(bi, 0.35) : "transparent";
    mmc.strokeStyle = on ? fieldColor(bi, 0.9) : CSS["--rule"];
    mmc.lineWidth = 1;
    const x = offX + (bl.x - b.x0) * k, y = offY + (bl.y - b.y0) * k;
    if (on) mmc.fillRect(x, y, bl.w * k, bl.h * k);
    mmc.strokeRect(x, y, bl.w * k, bl.h * k);
  });
  mmc.strokeStyle = CSS.result;
  mmc.lineWidth = 1.5;
  const vx = offX + (toWorldX(0) - b.x0) * k, vy = offY + (toWorldY(0) - b.y0) * k;
  mmc.strokeRect(vx, vy, (W / cam.s) * k, (H / cam.s) * k);
}
mm.addEventListener("pointerdown", (e) => {
  const r = mm.getBoundingClientRect(), b = mm.__b, k = mm.__k, [ox, oy] = mm.__off;
  cam.x = b.x0 + (e.clientX - r.left - ox) / k;
  cam.y = b.y0 + (e.clientY - r.top - oy) / k;
  draw();
});

/* =====================================================================
   7.  Hit testing
   ===================================================================== */
function paperAt(wx, wy) {
  for (let bi = 0; bi < BLOCKS.length; bi++) {
    const b = BLOCKS[bi];
    if (!showField[b.fieldIdx]) continue;
    if (wx < b.x || wy < b.y || wx > b.x + b.w || wy > b.y + b.h) continue;
    const c = Math.floor((wx - b.x) / PITCH_X), r = Math.floor((wy - b.y) / PITCH_Y);
    const k = r * b.cols + c;
    if (c < 0 || c >= b.cols || k < 0 || k >= b.members.length) return null;
    return b.members[k];
  }
  return null;
}

function nodeAt(sx, sy) {
  const wx = toWorldX(sx), wy = toWorldY(sy);
  const pi = paperAt(wx, wy);
  if (pi === null) return null;
  const cellPx = CELL_W * cam.s;
  const [ox, oy] = cellOrigin(pi);

  /* zoomed all the way out a cell is a single dot: hit the paper directly */
  if (cellPx < L_NODES) {
    const L = layoutOf(pi);
    return { pi, ni: L.nodes.findIndex((n) => n.t === "paper") };
  }

  const L = layoutOf(pi);
  const lx = wx - ox, ly = wy - oy;
  const boxes = cellPx >= L_BOXES;
  let best = null, bestD = Infinity;
  for (let i = 0; i < L.nodes.length; i++) {
    const n = L.nodes[i];
    if (!showType[n.t]) continue;
    if (boxes) {
      const h = 70;   // generous: real height varies with the wrapped line count
      if (lx >= lx_of(n) && lx <= rx_of(n) && ly >= n.y - h / 2 && ly <= n.y + h / 2)
        return { pi, ni: i };
    } else {
      const d = Math.hypot(lx - cx_of(n), ly - n.y);
      const r = Math.max(R_NODE[n.t], 16 / cam.s);
      if (d < r && d < bestD) { bestD = d; best = { pi, ni: i }; }
    }
  }
  return best;
}

/* =====================================================================
   8.  Interaction
   ===================================================================== */
let dragging = false, dragMoved = false, lastX = 0, lastY = 0;

cv.addEventListener("pointerdown", (e) => {
  dragging = true; dragMoved = false;
  lastX = e.clientX; lastY = e.clientY;
  cv.setPointerCapture(e.pointerId);
  cv.classList.add("dragging");
});
cv.addEventListener("pointermove", (e) => {
  if (dragging) {
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    if (Math.abs(dx) + Math.abs(dy) > 2) dragMoved = true;
    cam.x -= dx / cam.s; cam.y -= dy / cam.s;
    lastX = e.clientX; lastY = e.clientY;
    hideTip();
    draw();
    return;
  }
  const hit = nodeAt(e.clientX, e.clientY);
  const changed = JSON.stringify(hit) !== JSON.stringify(hover);
  hover = hit;
  if (hit) showTip(e.clientX, e.clientY, hit); else hideTip();
  if (changed) draw();
});
cv.addEventListener("pointerup", (e) => {
  dragging = false;
  cv.classList.remove("dragging");
  if (dragMoved) return;
  const hit = nodeAt(e.clientX, e.clientY);
  pinned = hit;
  if (hit) openDetail(hit); else closeDetail();
  draw();
});
cv.addEventListener("pointerleave", () => { hover = null; hideTip(); draw(); });

cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const wx = toWorldX(e.clientX), wy = toWorldY(e.clientY);
  const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0016));
  const b = worldBounds();
  const minS = Math.min(W / (b.x1 - b.x0), H / (b.y1 - b.y0)) * 0.55;
  cam.s = Math.max(minS, Math.min(3.5, cam.s * factor));
  /* keep the point under the cursor pinned to the cursor */
  cam.x = wx - (e.clientX - W / 2) / cam.s;
  cam.y = wy - (e.clientY - H / 2) / cam.s;
  hideTip();
  draw();
}, { passive: false });

addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") { if (e.key === "Escape") e.target.blur(); return; }
  const step = 120 / cam.s;
  if (e.key === "0") fit();
  else if (e.key === "+" || e.key === "=") { cam.s = Math.min(3.5, cam.s * 1.35); draw(); }
  else if (e.key === "-") { cam.s = cam.s / 1.35; draw(); }
  else if (e.key === "ArrowLeft") { cam.x -= step; draw(); }
  else if (e.key === "ArrowRight") { cam.x += step; draw(); }
  else if (e.key === "ArrowUp") { cam.y -= step; draw(); }
  else if (e.key === "ArrowDown") { cam.y += step; draw(); }
  else if (e.key === "Escape") { pinned = null; closeDetail(); draw(); }
  else if (e.key === "/") { e.preventDefault(); document.getElementById("search").focus(); }
});

document.getElementById("zIn").onclick = () => { cam.s = Math.min(3.5, cam.s * 1.5); draw(); };
document.getElementById("zOut").onclick = () => { cam.s = cam.s / 1.5; draw(); };
document.getElementById("fit").onclick = () => fit();

/* =====================================================================
   9.  Tooltip + detail panel
   ===================================================================== */
const tip = document.getElementById("tip");
function showTip(px, py, hit) {
  const L = layoutOf(hit.pi);
  const n = L.nodes[hit.ni];
  if (!n) return hideTip();
  const txt = TT.get(n.ti);
  tip.querySelector(".badge").textContent = n.t;
  tip.querySelector(".badge").style.color = CSS[n.t];
  tip.querySelector(".txt").textContent = txt.length > 420 ? txt.slice(0, 420) + "…" : txt;
  tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(px + 16, W - r.width - 10) + "px";
  tip.style.top = Math.min(py + 16, H - r.height - 10) + "px";
}
function hideTip() { tip.style.display = "none"; }

const detail = document.getElementById("detail");
function openDetail(hit) {
  const L = layoutOf(hit.pi);
  const n = L.nodes[hit.ni];
  if (!n) return;
  const s = S.papers[hit.pi];
  const nClaims = s[3].length;
  const nEv = s[3].reduce((a, c) => a + c[1].length, 0);   /* c = [local, evidence[], implication[]] */
  const nImp = s[3].reduce((a, c) => a + c[2].length, 0);

  const badge = document.getElementById("dBadge");
  badge.textContent = n.t;
  badge.style.color = CSS[n.t];
  badge.style.borderColor = CSS[n.t];
  badge.style.background = CSS[n.t + "-bg"];
  document.getElementById("dBody").textContent = TT.get(n.ti);

  const rows = [
    ["node id", `${n.t}_${n.local}`],
    ["field", S.fields[s[0]]],
    ["paper", `paper_${hit.pi}`],
  ];
  if (n.t !== "paper") rows.push(["paper title", TT.get(BASE.paper + hit.pi).split(". ")[0]]);
  rows.push(["subgraph", `${nClaims} claims · ${nEv} evidence · ${nImp} implications`]);
  document.getElementById("dMeta").innerHTML =
    rows.map(([k, v]) => `<div><span>${k}</span><span>${escHtml(v)}</span></div>`).join("");
  detail.classList.add("open");
}
function closeDetail() { detail.classList.remove("open"); pinned = null; }
document.getElementById("close").onclick = () => { closeDetail(); draw(); };
function escHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/* =====================================================================
   10. Search — titles only, so we decode just the paper slice up front
   ===================================================================== */
function buildTitles() {
  TITLES = new Array(S.papers.length);
  for (let i = 0; i < S.papers.length; i++) {
    const t = TT.get(BASE.paper + i);
    const cut = t.indexOf(". ");
    TITLES[i] = (cut > 12 ? t.slice(0, cut) : t.slice(0, 110));
  }
  TT.cache.clear();
}
const searchEl = document.getElementById("search"), resultsEl = document.getElementById("results");
searchEl.addEventListener("input", () => {
  const q = searchEl.value.trim().toLowerCase();
  resultsEl.innerHTML = "";
  if (q.length < 2) return;
  const hits = [];
  for (let i = 0; i < TITLES.length && hits.length < 25; i++)
    if (TITLES[i].toLowerCase().includes(q)) hits.push(i);
  resultsEl.innerHTML = hits.map((i) =>
    `<li data-i="${i}">${escHtml(TITLES[i].slice(0, 96))}` +
    `<div class="rf">${escHtml(S.fields[S.papers[i][0]])} · paper_${i}</div></li>`).join("");
});
resultsEl.addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li) return;
  const pi = +li.dataset.i;
  showField[S.papers[pi][0]] = true;
  syncToggles();
  flyToPaper(pi);
  const L = layoutOf(pi);
  pinned = { pi, ni: L.nodes.findIndex((n) => n.t === "paper") };
  openDetail(pinned);
  draw();
});

/* =====================================================================
   11. Toggle rails
   ===================================================================== */
function toggleRow(container, checked, swatchHtml, label, count, onChange, gloss) {
  const li = document.createElement("li");
  li.innerHTML = `<input type="checkbox" ${checked ? "checked" : ""}>${swatchHtml}` +
                 `<code>${escHtml(label)}</code>` +
                 (gloss ? `<span class="gloss">${escHtml(gloss)}</span>` : "") +
                 (count !== null ? `<span class="count">${count.toLocaleString()}</span>` : "");
  const box = li.querySelector("input");
  li.onclick = (e) => {
    if (e.target !== box) box.checked = !box.checked;
    onChange(box.checked);
    draw();
  };
  container.appendChild(li);
  return box;
}
const toggleBoxes = { field: {}, type: {}, rel: {} };

function buildRails() {
  const st = META.stats;
  document.getElementById("summary").textContent =
    `${st.total_nodes.toLocaleString()} nodes · ${st.total_edges.toLocaleString()} edges · ` +
    `${st.coverage.num_papers.toLocaleString()} papers`;

  const fc = document.getElementById("fieldToggles");
  BLOCKS.forEach((b, bi) => {
    showField[b.fieldIdx] = true;
    toggleBoxes.field[b.fieldIdx] = toggleRow(
      fc, true,
      `<span class="dot" style="background:${fieldColor(bi, 0.35)};border-color:${fieldColor(bi, 1)}"></span>`,
      b.name, b.members.length, (v) => { showField[b.fieldIdx] = v; }, null);
  });

  const tc = document.getElementById("typeToggles");
  for (const t of META.nodeTypes) {
    if (t === "field") continue;              // hubs are shown as block headers
    showType[t] = true;
    toggleBoxes.type[t] = toggleRow(
      tc, true,
      `<span class="dot" style="background:var(${META.typeColor[t]}-bg);border-color:var(${META.typeColor[t]})"></span>`,
      t, st.nodes[t] ?? 0, (v) => { showType[t] = v; }, null);
  }

  const rc = document.getElementById("relToggles");
  for (const rel in META.relStyle) {
    const [varName, dashed, gloss] = META.relStyle[rel];
    /* `in_field` is not drawn as edges at all — it IS the block grouping, and
       6.2k lines converging on five points would be noise, not signal. */
    if (rel === "in_field") { showRel[rel] = false; continue; }
    showRel[rel] = true;
    const style = dashed ? "dashed" : "solid";
    const n = st.edges[Object.keys(st.edges).find((k) => k.split("__")[1] === rel)] ?? 0;
    toggleBoxes.rel[rel] = toggleRow(
      rc, true,
      `<span class="bar" style="border-top:2px ${style} var(${varName})"></span>`,
      rel, n, (v) => { showRel[rel] = v; }, null);
    toggleBoxes.rel[rel].parentElement.title = gloss;
  }
  const note = document.createElement("p");
  note.className = "sub";
  note.style.marginTop = "8px";
  note.textContent = `in_field is drawn as the block grouping itself. ` +
    `Of ${S.cites.length} cites edges, ${S.selfCites} are self-loops — ` +
    `only ${S.cites.length - S.selfCites} actually join two papers.`;
  rc.parentElement.appendChild(note);
}
function syncToggles() {
  for (const k in toggleBoxes.field) toggleBoxes.field[k].checked = showField[k];
}

function updateHud(nVisible) {
  const cellPx = CELL_W * cam.s;
  const lod = cellPx < L_NODES ? "papers" : cellPx < L_EDGES ? "nodes"
            : cellPx < L_BOXES ? "edges" : "text";
  document.getElementById("zoomlvl").textContent = `${lod} · ${nVisible}`;
}

/* =====================================================================
   12. Boot
   ===================================================================== */
function resize() {
  DPR = Math.min(2, devicePixelRatio || 1);
  W = innerWidth; H = innerHeight;
  cv.width = W * DPR; cv.height = H * DPR;
  cv.style.width = W + "px"; cv.style.height = H + "px";
  draw();
}
addEventListener("resize", resize);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readTheme(); draw(); });

(async function boot() {
  const loading = document.getElementById("loading");
  if (typeof DecompressionStream === "undefined") {
    loading.textContent = "This browser lacks DecompressionStream — try Chrome 80+, Safari 16.4+ or Firefox 113+.";
    return;
  }
  try {
    S = JSON.parse(new TextDecoder().decode(await gunzip(b64ToBytes(STRUCT_B64))));
    TT = new TextTable(await gunzip(b64ToBytes(TEXT_B64)));
    BASE = S.bases;
    buildBlocks();
    buildTitles();
    readTheme();
    buildRails();
    resize();
    fit();
    loading.remove();
  } catch (err) {
    loading.textContent = "Failed to load graph: " + err.message;
    console.error(err);
  }
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", default=os.path.join(HERE, "data", "kg_health_sciences"),
                    help="directory holding nodes.json / edges.json / stats.json")
    ap.add_argument("--out", default=None, help="default: <kg>/explorer.html")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.kg, "explorer.html")
    build_page(args.kg, out, args.title)


if __name__ == "__main__":
    main()
