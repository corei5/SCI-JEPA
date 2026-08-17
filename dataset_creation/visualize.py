"""
Render a sample knowledge graph (from build_sample.py) as a standalone HTML page.

Two figures, both drawn as hand-authored inline SVG — no JS, no libraries, no
external assets, so the file opens straight from disk:

  1. SCHEMA      the 7 node types and 10 relations, generated from schema.EDGE_TYPES
                 so it cannot drift from what the builder actually emits.
  2. INSTANCE    the real sample graph, one horizontal band per paper, laid out
                 along the reasoning chain
                     method -> result -> claim -> {evidence, implication}

Usage
-----
    python -m dataset_creation.build_sample --snippet 400
    python -m dataset_creation.visualize            # -> sample/sample_kg.html
"""

import argparse
import html
import json
import os

from .schema import EDGE_TYPES, NODE_TYPES

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAMPLE = os.path.join(HERE, "sample")

# ---------------------------------------------------------------------------
# Column geometry for the instance figure: x offset and box width per node type.
# ---------------------------------------------------------------------------
# x offsets leave a LEFT_GUTTER-wide margin: paper->paper `cites` edges join two
# boxes in the SAME column, so they are routed around the left rather than drawn
# straight back through the column, where the node boxes would hide them.
LEFT_GUTTER = 60
COLS = {
    "field":       (80, 110),
    "paper":       (220, 210),
    "method":      (460, 180),
    "result":      (670, 180),
    "claim":       (880, 230),
    "evidence":    (1140, 300),
    "implication": (1140, 300),
}
CANVAS_W = 1460
LEAF_H = 54          # vertical pitch of one evidence/implication row
PAPER_GAP = 44       # blank space between paper bands
BOX_H = 42
LINE_H = 13
CHAR_PX = 5.9        # rough advance width at 11px in the page's sans stack

# Node fill/stroke, as CSS custom properties defined once in the page.
TYPE_COLOR = {
    "paper":       "--c-paper",
    "field":       "--c-field",
    "claim":       "--c-claim",
    "method":      "--c-method",
    "result":      "--c-result",
    "evidence":    "--c-evidence",
    "implication": "--c-implication",
}

# Relation -> (stroke var, dashed?, human gloss for the legend).
REL_STYLE = {
    "has_claim":     ("--e-struct", False, "paper owns this claim"),
    "has_method":    ("--e-struct", False, "paper owns this method"),
    "has_result":    ("--e-struct", False, "paper owns this result"),
    "in_field":      ("--e-struct", True,  "paper joins a shared field hub"),
    "cites":         ("--e-cites",  False, "intra-corpus citation"),
    "produces":      ("--e-chain",  False, "imposed: method yields result"),
    "grounds":       ("--e-chain",  False, "imposed: result backs every claim"),
    "supported_by":  ("--e-support", False, "from supporting_evidence"),
    "challenged_by": ("--e-contra", False, "from contradicting_evidence"),
    "implies":       ("--e-implies", True, "from claim.implications"),
}
REL_ORDER = [r for _, r, _, _ in EDGE_TYPES]


# ===========================================================================
#  text helpers
# ===========================================================================
def wrap(text, width_px, max_lines):
    """Greedy word wrap into at most `max_lines`, ellipsising the overflow."""
    budget = max(4, int((width_px - 16) / CHAR_PX))
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if len(cand) <= budget:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
            if len(lines) == max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if not lines:
        return [""]
    if len(lines) == max_lines and (len(words) > sum(len(l.split()) for l in lines)):
        lines[-1] = lines[-1][:budget - 1].rstrip() + "…"
    return lines


def esc(s):
    return html.escape(str(s), quote=True)


# ===========================================================================
#  SVG primitives
# ===========================================================================
def node_box(x, y, w, h, lines, color_var, title, node_id):
    """A rounded rect + wrapped label + <title> tooltip carrying the full text."""
    top = y - h / 2
    first = top + (h - len(lines) * LINE_H) / 2 + LINE_H - 3
    out = [f'<g><title>{esc(node_id)} — {esc(title)}</title>',
           f'<rect x="{x}" y="{top:.1f}" width="{w}" height="{h}" rx="6" '
           f'fill="var({color_var}-bg)" stroke="var({color_var})" stroke-width="1.25"/>']
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x + 8}" y="{first + i * LINE_H:.1f}" '
                   f'font-size="11">{esc(ln)}</text>')
    out.append("</g>")
    return "".join(out)


def edge_path(x1, y1, x2, y2, stroke_var, dashed, marker, label=None):
    """Cubic bezier between two anchor points, flowing horizontally."""
    dx = max(30.0, abs(x2 - x1) * 0.42)
    if x2 >= x1:
        d = f"M {x1:.1f} {y1:.1f} C {x1 + dx:.1f} {y1:.1f}, {x2 - dx:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}"
    else:
        d = f"M {x1:.1f} {y1:.1f} C {x1 - dx:.1f} {y1:.1f}, {x2 + dx:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}"
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out = [f'<path d="{d}" fill="none" stroke="var({stroke_var})" stroke-width="1.3"'
           f'{dash} marker-end="url(#{marker})" opacity="0.85"/>']
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 5
        out.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="11" text-anchor="middle" '
                   f'fill="var({stroke_var})">{esc(label)}</text>')
    return "".join(out)


def markers(prefix):
    """One arrowhead per stroke colour. `prefix` keeps the two figures' ids apart."""
    seen, out = [], []
    for var, _, _ in REL_STYLE.values():
        if var in seen:
            continue
        seen.append(var)
        out.append(f'<marker id="{marker_id(prefix, var)}" viewBox="0 0 8 8" refX="7" '
                   f'refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                   f'<path d="M 0 0 L 8 4 L 0 8 z" fill="var({var})"/></marker>')
    return "<defs>" + "".join(out) + "</defs>"


def marker_id(prefix, var):
    return f"{prefix}-arw" + var.replace("--", "").replace("-", "")


# ===========================================================================
#  Figure 1 — schema
# ===========================================================================
SCHEMA_POS = {
    "field":       (60, 60),
    "paper":       (60, 190),
    "method":      (300, 190),
    "result":      (520, 190),
    "claim":       (740, 190),
    "evidence":    (980, 110),
    "implication": (980, 275),
}
SCHEMA_W, SCHEMA_H = 150, 40


def schema_svg():
    """The 7 node types and every relation the builder can emit."""
    parts = [markers("sc")]

    def anchor(nt, side):
        x, y = SCHEMA_POS[nt]
        return (x + SCHEMA_W, y) if side == "r" else (x, y)

    # curve the paper->claim / paper->result edges below the main chain so they
    # do not run through the method/result boxes
    below = {"has_claim": 285, "has_result": 250}

    for src, rel, dst, _prop in EDGE_TYPES:
        var, dashed, _ = REL_STYLE[rel]
        mk = marker_id("sc", var)
        if rel == "cites":                                  # self-loop on paper
            x, y = SCHEMA_POS["paper"]
            parts.append(
                f'<path d="M {x + 30} {y - 20} C {x + 10} {y - 70}, {x + 120} {y - 70}, '
                f'{x + 100} {y - 20}" fill="none" stroke="var({var})" stroke-width="1.3" '
                f'marker-end="url(#{mk})"/>'
                f'<text x="{x + 65}" y="{y - 54}" font-size="11" text-anchor="middle" '
                f'fill="var({var})">cites</text>')
            continue
        if rel == "in_field":
            x, y = SCHEMA_POS["paper"]
            fx, fy = SCHEMA_POS["field"]
            parts.append(
                f'<path d="M {x + 75} {y - 20} L {fx + 75} {fy + 20}" fill="none" '
                f'stroke="var({var})" stroke-width="1.3" stroke-dasharray="5 4" '
                f'marker-end="url(#{mk})"/>'
                f'<text x="{x + 82}" y="{(y + fy) / 2:.0f}" font-size="11" '
                f'fill="var({var})">in_field</text>')
            continue
        if rel in below:
            sy = SCHEMA_POS[src][1]
            dx_, dy_ = SCHEMA_POS[dst][0] + SCHEMA_W / 2, SCHEMA_POS[dst][1] + 20
            yy = below[rel]
            parts.append(
                f'<path d="M {SCHEMA_POS[src][0] + SCHEMA_W / 2:.0f} {sy + 20} '
                f'L {SCHEMA_POS[src][0] + SCHEMA_W / 2:.0f} {yy} L {dx_:.0f} {yy} L {dx_:.0f} {dy_}" '
                f'fill="none" stroke="var({var})" stroke-width="1.3" '
                f'marker-end="url(#{mk})"/>'
                f'<text x="{(SCHEMA_POS[src][0] + SCHEMA_W / 2 + dx_) / 2:.0f}" y="{yy - 6}" '
                f'font-size="11" text-anchor="middle" fill="var({var})">{rel}</text>')
            continue
        sx, sy = anchor(src, "r")
        dx_, dy_ = anchor(dst, "l")
        parts.append(edge_path(sx, sy, dx_, dy_, var, dashed, mk, rel))

    for nt, (x, y) in SCHEMA_POS.items():
        parts.append(
            f'<rect x="{x}" y="{y - 20}" width="{SCHEMA_W}" height="{SCHEMA_H}" rx="6" '
            f'fill="var({TYPE_COLOR[nt]}-bg)" stroke="var({TYPE_COLOR[nt]})" stroke-width="1.5"/>'
            f'<text x="{x + SCHEMA_W / 2}" y="{y + 4}" font-size="13" text-anchor="middle" '
            f'font-weight="600">{nt}</text>')

    return (f'<svg viewBox="0 0 1180 330" role="img" aria-label="Schema of the paper '
            f'knowledge graph: paper nodes own method, result and claim nodes; method '
            f'produces result, result grounds claim, and each claim links to supporting '
            f'evidence, contradicting evidence and implications. Papers also link to a '
            f'shared field hub and cite other papers.">'
            + "".join(parts) + "</svg>")


# ===========================================================================
#  Figure 2 — the actual sample instance
# ===========================================================================
def instance_svg(nodes, edges):
    by_id = {n["id"]: n for n in nodes}
    out_edges = {}
    for e in edges:
        out_edges.setdefault(e["rel"], []).append(e)

    # claim -> its leaves, in the order the builder created them
    leaves_of = {}
    for rel in ("supported_by", "challenged_by", "implies"):
        for e in out_edges.get(rel, []):
            leaves_of.setdefault(e["src"], []).append(e["dst"])

    claims_of, method_of, result_of = {}, {}, {}
    for e in out_edges.get("has_claim", []):
        claims_of.setdefault(e["src"], []).append(e["dst"])
    for e in out_edges.get("has_method", []):
        method_of[e["src"]] = e["dst"]
    for e in out_edges.get("has_result", []):
        result_of[e["src"]] = e["dst"]

    papers = [n["id"] for n in nodes if n["type"] == "paper"]

    # ---- vertical layout: leaves drive everything, parents centre on children
    y = {}
    cursor = 40.0
    for p in papers:
        claim_ys = []
        for c in claims_of.get(p, []):
            lv = leaves_of.get(c, [])
            if lv:
                for leaf in lv:
                    y[leaf] = cursor
                    cursor += LEAF_H
                claim_ys.append(sum(y[l] for l in lv) / len(lv))
            else:
                claim_ys.append(cursor)
                cursor += LEAF_H
            y[c] = claim_ys[-1]
        band = sum(claim_ys) / len(claim_ys) if claim_ys else cursor
        y[p] = band
        if p in method_of:
            y[method_of[p]] = band - 26
        if p in result_of:
            y[result_of[p]] = band + 26
        cursor += PAPER_GAP

    for n in nodes:                       # field hubs centre on their papers
        if n["type"] == "field":
            members = [y[e["src"]] for e in out_edges.get("in_field", [])
                       if e["dst"] == n["id"]]
            y[n["id"]] = sum(members) / len(members) if members else cursor / 2

    height = cursor + 40

    # ---- edges first so boxes sit on top
    parts = [markers("in")]
    for e in edges:
        s, d = by_id[e["src"]], by_id[e["dst"]]
        var, dashed, _ = REL_STYLE[e["rel"]]
        mk = marker_id("in", var)
        sx0, sw = COLS[s["type"]]
        dx0, dw = COLS[d["type"]]
        if e["src"] == e["dst"]:                       # self-citation loop
            yy = y[e["src"]]
            parts.append(
                f'<path d="M {sx0 + 40} {yy - 22} C {sx0 + 20} {yy - 60}, '
                f'{sx0 + 150} {yy - 60}, {sx0 + 130} {yy - 22}" fill="none" '
                f'stroke="var({var})" stroke-width="1.3" marker-end="url(#{mk})"/>'
                f'<text x="{sx0 + 85}" y="{yy - 48}" font-size="10" text-anchor="middle" '
                f'fill="var({var})">cites (self)</text>')
            continue
        if sx0 == dx0:                                 # same column (paper -> paper)
            ys, yd = y[e["src"]], y[e["dst"]]
            gx = LEFT_GUTTER / 2                       # route around the left margin
            parts.append(
                f'<path d="M {sx0} {ys:.1f} C {gx:.0f} {ys:.1f}, {gx:.0f} {yd:.1f}, '
                f'{dx0} {yd:.1f}" fill="none" stroke="var({var})" stroke-width="1.5" '
                f'marker-end="url(#{mk})"/>'
                f'<text x="{gx + 6:.0f}" y="{(ys + yd) / 2:.1f}" font-size="10" '
                f'fill="var({var})" transform="rotate(-90 {gx + 6:.0f} '
                f'{(ys + yd) / 2:.1f})" text-anchor="middle">cites</text>')
        elif dx0 < sx0:                                # right-to-left (in_field)
            parts.append(edge_path(sx0, y[e["src"]], dx0 + dw, y[e["dst"]], var, dashed, mk))
        else:
            parts.append(edge_path(sx0 + sw, y[e["src"]], dx0, y[e["dst"]], var, dashed, mk))

    # ---- boxes
    for n in nodes:
        x0, w = COLS[n["type"]]
        lines = wrap(n["text"], w, 3 if n["type"] in ("paper", "claim") else 2)
        h = max(BOX_H, len(lines) * LINE_H + 16)
        parts.append(node_box(x0, y[n["id"]], w, h, lines,
                              TYPE_COLOR[n["type"]], n["text"][:220], n["id"]))

    # ---- column headings
    seen = set()
    for nt in NODE_TYPES:
        x0, w = COLS[nt]
        key = (x0, w)
        label = "evidence / implication" if nt in ("evidence", "implication") else nt
        if key in seen:
            continue
        seen.add(key)
        parts.append(f'<text x="{x0}" y="20" font-size="12" font-weight="600" '
                     f'opacity="0.65">{esc(label)}</text>')

    return (f'<svg viewBox="0 0 {CANVAS_W} {height:.0f}" role="img" aria-label="The sample '
            f'knowledge graph: three medical papers, each expanded into its own reasoning '
            f'subgraph of method, result, claims, supporting and contradicting evidence and '
            f'implications, all sharing one Medicine field hub.">'
            + "".join(parts) + "</svg>")


# ===========================================================================
#  Page
# ===========================================================================
PALETTE_LIGHT = {
    "--c-paper": "#4338ca", "--c-paper-bg": "#eef2ff",
    "--c-field": "#475569", "--c-field-bg": "#f1f5f9",
    "--c-claim": "#7c3aed", "--c-claim-bg": "#f5f3ff",
    "--c-method": "#0f766e", "--c-method-bg": "#ecfdf5",
    "--c-result": "#b45309", "--c-result-bg": "#fffbeb",
    "--c-evidence": "#0369a1", "--c-evidence-bg": "#f0f9ff",
    "--c-implication": "#be185d", "--c-implication-bg": "#fdf2f8",
    "--e-struct": "#94a3b8", "--e-chain": "#4338ca", "--e-support": "#15803d",
    "--e-contra": "#dc2626", "--e-implies": "#be185d", "--e-cites": "#b45309",
    "--bg": "#ffffff", "--fg": "#0f172a", "--muted": "#64748b",
    "--rule": "#e2e8f0", "--panel": "#f8fafc",
}
PALETTE_DARK = {
    "--c-paper": "#a5b4fc", "--c-paper-bg": "#1e1b4b",
    "--c-field": "#cbd5e1", "--c-field-bg": "#1e293b",
    "--c-claim": "#c4b5fd", "--c-claim-bg": "#2e1065",
    "--c-method": "#5eead4", "--c-method-bg": "#042f2e",
    "--c-result": "#fcd34d", "--c-result-bg": "#3b2506",
    "--c-evidence": "#7dd3fc", "--c-evidence-bg": "#082f49",
    "--c-implication": "#f9a8d4", "--c-implication-bg": "#4c0519",
    "--e-struct": "#64748b", "--e-chain": "#a5b4fc", "--e-support": "#4ade80",
    "--e-contra": "#f87171", "--e-implies": "#f9a8d4", "--e-cites": "#fcd34d",
    "--bg": "#0b1120", "--fg": "#e2e8f0", "--muted": "#94a3b8",
    "--rule": "#1e293b", "--panel": "#111c33",
}


def _vars(p):
    return "\n".join(f"  {k}: {v};" for k, v in p.items())


def legend_html():
    rows = []
    for rel in REL_ORDER:
        var, dashed, gloss = REL_STYLE[rel]
        style = "dashed" if dashed else "solid"
        rows.append(
            f'<li><span class="swatch" style="border-top:2px {style} var({var})"></span>'
            f'<code>{esc(rel)}</code><span class="gloss">{esc(gloss)}</span></li>')
    types = "".join(
        f'<li><span class="dot" style="background:var({TYPE_COLOR[nt]}-bg);'
        f'border-color:var({TYPE_COLOR[nt]})"></span><code>{nt}</code></li>'
        for nt in NODE_TYPES)
    return (f'<div class="legends"><div><h3>Node types</h3><ul class="lg types">{types}</ul></div>'
            f'<div><h3>Relations</h3><ul class="lg">{"".join(rows)}</ul></div></div>')


def stats_html(stats):
    cov = stats["coverage"]
    nodes = "".join(f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
                    for k, v in stats["nodes"].items())
    edges = "".join(
        f'<tr><td><code>{k.replace("__", ", ")}</code></td><td>{v}</td></tr>'
        for k, v in stats["edges"].items())
    return (
        f'<div class="tables">'
        f'<div><h3>Nodes ({stats["total_nodes"]})</h3><table>{nodes}</table></div>'
        f'<div><h3>Edges ({stats["total_edges"]})</h3><table>{edges}</table></div>'
        f'</div>'
        f'<p class="note">Parsed {cov["num_papers"]} records, skipped {cov["skipped"]}. '
        f'{cov["papers_with_claims"]} carried claims, {cov["papers_with_method"]} a method, '
        f'{cov["papers_with_result"]} a result. '
        f'{stats["dangling_citations_dropped"]} referenced works pointed outside the corpus '
        f'and were dropped.</p>')


def build_page(sample_dir, out_path, title="Sample paper knowledge graph"):
    nodes = json.load(open(os.path.join(sample_dir, "nodes.json"), encoding="utf-8"))
    edges = json.load(open(os.path.join(sample_dir, "edges.json"), encoding="utf-8"))
    stats = json.load(open(os.path.join(sample_dir, "stats.json"), encoding="utf-8"))

    page = f"""<title>{esc(title)}</title>
<style>
:root {{
{_vars(PALETTE_LIGHT)}
}}
:root:not([data-theme="light"]) {{ }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
{_vars(PALETTE_DARK)}
  }}
}}
:root[data-theme="dark"] {{
{_vars(PALETTE_DARK)}
}}
body {{ background: var(--bg); color: var(--fg); margin: 0;
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 40px 24px 80px; }}
h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }}
h2 {{ font-size: 19px; margin: 44px 0 10px; padding-top: 18px;
  border-top: 1px solid var(--rule); }}
h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); margin: 0 0 8px; }}
p {{ color: var(--fg); max-width: 72ch; }}
.sub {{ color: var(--muted); margin-top: 0; }}
figure {{ margin: 18px 0 8px; padding: 14px; background: var(--panel);
  border: 1px solid var(--rule); border-radius: 10px; overflow-x: auto; }}
figure svg {{ display: block; max-width: 100%; height: auto; color: var(--fg); }}
figure svg text {{ fill: var(--fg); font-family: inherit; }}
figcaption {{ color: var(--muted); font-size: 13px; margin-top: 10px; max-width: 72ch; }}
.tall svg {{ max-width: none; width: {CANVAS_W}px; }}
.legends {{ display: flex; gap: 40px; flex-wrap: wrap; margin: 16px 0; }}
ul.lg {{ list-style: none; padding: 0; margin: 0; font-size: 13px; }}
ul.lg li {{ display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }}
.swatch {{ width: 26px; height: 0; flex: none; }}
.dot {{ width: 13px; height: 13px; border-radius: 3px; border: 1.5px solid; flex: none; }}
.gloss {{ color: var(--muted); }}
code {{ font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace; }}
.tables {{ display: flex; gap: 40px; flex-wrap: wrap; margin: 14px 0; }}
table {{ border-collapse: collapse; font-size: 13px; }}
td {{ padding: 3px 18px 3px 0; border-bottom: 1px solid var(--rule); }}
td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
.note {{ color: var(--muted); font-size: 13px; }}
</style>

<div class="wrap">
<h1>{esc(title)}</h1>
<p class="sub">Built by <code>dataset_creation/build_sample.py</code> from
{stats["coverage"]["num_papers"]} raw paper records — the same parsing and wiring the
full-corpus graph uses, minus the MiniLM embedding pass.</p>

<h2>Schema</h2>
<p>Every raw record becomes one <code>paper</code> node plus a small reasoning
subgraph hanging off it. Two of the relations are <em>imposed</em> rather than read
from the data: <code>produces</code> and <code>grounds</code> wire a paper's method to
its result and its result to every one of its claims, giving each paper an internal
chain instead of a flat star.</p>
<figure>
{schema_svg()}
<figcaption>Node types and the ten relations <code>build_hetero_graph()</code> emits.
Solid grey edges are ownership, indigo is the imposed method→result→claim chain,
green and red are the two evidence polarities.</figcaption>
</figure>
{legend_html()}

<h2>This sample</h2>
{stats_html(stats)}

<h2>The graph</h2>
<p>One horizontal band per paper. Reading left to right follows the reasoning chain;
hover any box for its full text and node id.</p>
<figure class="tall">
{instance_svg(nodes, edges)}
<figcaption>{stats["total_nodes"]} nodes and {stats["total_edges"]} edges from
{stats["coverage"]["num_papers"]} papers. All three share one
<code>{esc(stats["fields"][0])}</code> field hub — that hub and the directed
<code>cites</code> relation are the only things connecting one paper's subgraph to
another's, and this sample has {stats["citations"]["edges"]} citation edges
({stats["citations"]["dangling_dropped"]} references point outside the corpus,
{stats["citations"]["self_citations_dropped"]} were self-citations).</figcaption>
</figure>
</div>
"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"[visualize] wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", default=DEFAULT_SAMPLE)
    ap.add_argument("--out", default=None, help="default: <sample-dir>/sample_kg.html")
    ap.add_argument("--title", default="Sample paper knowledge graph",
                    help="page <title>; give each build its own so pages stay "
                         "distinguishable")
    args = ap.parse_args()
    out = args.out or os.path.join(args.sample_dir, "sample_kg.html")
    build_page(args.sample_dir, out, title=args.title)


if __name__ == "__main__":
    main()
