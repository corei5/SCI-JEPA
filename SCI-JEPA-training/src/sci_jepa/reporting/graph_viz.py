"""
Animated, interactive HTML views of the KU graph (vis-network).

  paper_html     one paper's subgraph: hub + KUs + schema edges, built up node by node
  instance_html  one training instance, step by step: neighbourhood -> mask the
                 target -> hide what paraphrases it -> hide probe edges
  corpus_html    many papers: hubs appear one by one, linked by citations, proxy
                 grounded edges and shared subfields; optional KU level

Every page is self-contained except the vis-network script, which is loaded
from ./vendor (downloaded once by serve.ensure_vendor) with a CDN fallback.
Each page has play / pause / step / reset controls, a caption per stage,
physics on/off, and hover tooltips with the KU text.
"""
import html
import json
import os

import numpy as np

from sci_jepa.data.graph_index import HUB
from sci_jepa.data.ku_schema import COMP_TYPES, REL_ID, ROLE_RELS
from sci_jepa.dashboard.telemetry import REL_NAMES

# one colour per KU type (colour-blind-safe-ish, readable in light and dark)
TYPE_COLOR = {
    "context": "#8c6bb1", "question": "#6a51a3", "method": "#2171b5", "procedure": "#6baed6",
    "result": "#238b45", "interpretation": "#74c476", "limitation": "#d94801",
    "claim": "#e7298a", "evidence": "#fdae6b", "implication": "#c994c7", "paper": "#525252",
    "figures": "#969696", "robustness": "#969696", "data_code": "#969696", "ethics": "#969696",
    "hub": "#252525",
}
REL_COLOR = {"supported_by": "#31a354", "challenged_by": "#de2d26", "implies": "#c994c7",
             "grounds": "#e7298a", "produces": "#2171b5", "implemented_by": "#6baed6",
             "motivates": "#8c6bb1", "addressed_by": "#6a51a3", "interpreted_by": "#74c476",
             "limited_by": "#d94801", "proxy": "#ff7f00", "cites": "#000000",
             "same_subfield": "#bdbdbd", "same_field": "#e0e0e0", "member": "#d9d9d9"}


def _snip(gi, c, n=220):
    s = getattr(gi, "comp_snippet", None)
    return html.escape(str(s[c])[:n]) if s is not None else ""


def _ku_node(gi, c, nid, show_from=0, **kw):
    t = COMP_TYPES[gi.comp_type[c]]
    return {"id": nid, "label": t, "group": t, "color": TYPE_COLOR.get(t, "#999"),
            "title": f"<b>{t}</b><br>{_snip(gi, c)}", "shape": "dot", "size": 14,
            "show_from": show_from, **kw}


def short_id(pid):
    """arXiv ids stay as they are; file-path ids (titled / arxiv-old) -> file stem."""
    s = str(pid)
    if "/" in s and not s.split("/")[0].replace("-", "").replace(".", "").isalpha():
        s = os.path.splitext(s.rstrip("/").split("/")[-1])[0]
    return s[:14]


def _hub_node(gi, p, nid, show_from=0, **kw):
    f = gi.field_names[gi.paper_field[p]]
    return {"id": nid, "label": f"paper\n{short_id(gi.paper_ids[p])}", "group": "hub", "color": TYPE_COLOR["hub"],
            "title": f"<b>paper hub</b><br>{html.escape(str(gi.paper_ids[p]))}<br>field: {html.escape(f)}"
                     f"<br>year: {gi.paper_year[p]}", "shape": "box", "font": {"color": "#fff"},
            "show_from": show_from, **kw}


def _edge(a, b, rel, show_from=0, label=True, **kw):
    return {"from": a, "to": b, "title": rel,
            "label": rel if label and rel not in ("member", "same_subfield", "same_field") else "",
            "color": REL_COLOR.get(rel, "#999"), "rel": rel, "show_from": show_from,
            "arrows": "to" if rel in ROLE_RELS else "", "font": {"size": 9}, **kw}


# ---------------------------------------------------------------- views
def paper_html(gi, p, path):
    comps = [int(c) for c in gi.paper_comps[p]]
    order = sorted(comps, key=lambda c: (gi.comp_type[c], c))   # sections first, then claims...
    nodes = [_hub_node(gi, p, "hub", 0)]
    pos = {}
    for k, c in enumerate(order, 1):
        nodes.append(_ku_node(gi, c, f"k{c}", k)); pos[c] = k
    edges = [_edge("hub", f"k{c}", "member", pos[c]) for c in order]
    for c in order:
        for d, r in gi.comp_edges.get(c, []):
            if d in pos and c < d and 3 <= r < 3 + len(ROLE_RELS):
                rel = ROLE_RELS[r - 3]
                a, b = (c, d) if _src_first(gi, c, d, rel) else (d, c)
                edges.append(_edge(f"k{a}", f"k{b}", rel, max(pos[c], pos[d]), width=2))
    stages = [f"paper hub ({gi.field_names[gi.paper_field[p]]}, {gi.paper_year[p]})"] + \
             [f"+ {COMP_TYPES[gi.comp_type[c]]}" for c in order]
    _write(path, f"Paper subgraph: {gi.paper_ids[p]}", nodes, edges, stages,
           f"{len(comps)} knowledge units of one paper. Edges are imposed by the KU schema "
           f"(context→question→method→result→claim→evidence/implication). Hover a node for its text.",
           autoplay_ms=350)


def _src_first(gi, c, d, rel):
    from sci_jepa.data.ku_schema import SCHEMA_EDGES
    for s, r, t in SCHEMA_EDGES:
        if r == rel:
            return COMP_TYPES[gi.comp_type[c]] == s
    return True


def instance_html(gi, inst, cfg, path):
    """Animate how one training instance is built and masked."""
    from sci_jepa.data.ku_schema import PARAPHRASE_SIBLINGS
    anchor, papers = int(inst["target_comp"]), [int(p) for p in inst["papers"]]
    p0 = papers[0]
    vis = {int(c) for c in inst["ctx"]["comp"] if c >= 0} | {anchor}
    hidden = set(gi.children.get(anchor, ()))
    if "siblings" in cfg.mask_policy:
        sib = {COMP_TYPES.index(t) for t in PARAPHRASE_SIBLINGS.get(COMP_TYPES[gi.comp_type[anchor]], ())}
        hidden |= {int(c) for c in gi.paper_comps[p0] if gi.comp_type[c] in sib}
    allc = vis | hidden
    S_NB, S_MASK, S_DROP, S_PROBE = 1, 2, 3, 4
    cid_ = inst["ctx"]["comp"]
    probe_pairs = {frozenset((int(cid_[i]), int(cid_[j]))) for (i, j) in inst["probes"]}
    nodes, edges = [], []
    for k, p in enumerate(papers):
        nodes.append(_hub_node(gi, p, f"h{p}", 0 if k == 0 else S_NB))
    for c in sorted(allc):
        p = int(gi.comp_paper[c]); st = 0 if p == p0 else S_NB
        n = _ku_node(gi, c, f"k{c}", st, size=16 if p == p0 else 9)
        if p != p0:
            n["label"] = ""                          # neighbour KUs: hover for type + text
        if c == anchor:
            n["style_from"] = [[S_MASK, {"color": "#ff0000", "shape": "star", "size": 26, "label": "MASKED",
                                         "title": f"<b>masked target ({COMP_TYPES[gi.comp_type[c]]})</b><br>"
                                                  f"text hidden from the encoder:<br>{_snip(gi, c)}"}]]
        if c in hidden:
            n["style_from"] = [[S_DROP, {"color": "#e0e0e0", "label": "hidden", "opacity": 0.35}]]
        nodes.append(n)
        edges.append(_edge(f"h{p}", f"k{c}", "member", st))
    for c in sorted(allc):
        for d, r in gi.comp_edges.get(c, []):
            if d in allc and c < d and 3 <= r < 3 + len(ROLE_RELS):
                rel = ROLE_RELS[r - 3]
                a, b = (c, d) if _src_first(gi, c, d, rel) else (d, c)
                st = 0 if gi.comp_paper[c] == p0 and gi.comp_paper[d] == p0 else S_NB
                e = _edge(f"k{a}", f"k{b}", rel, st, label=(st == 0), width=2 if st == 0 else 1)
                if anchor in (c, d):
                    e["style_from"] = [[S_MASK, {"dashes": True, "color": "#ff9999", "label": "cut"}]]
                if c in hidden or d in hidden:
                    e["style_from"] = e.get("style_from", []) + [[S_DROP, {"hidden": True}]]
                if frozenset((c, d)) in probe_pairs:            # the encoder no longer sees it
                    e["style_from"] = e.get("style_from", []) + [[S_PROBE, {"hidden": True}]]
                edges.append(e)
    rel = inst["ctx"]["rel"]; hubs = [i for i, t in enumerate(inst["ctx"]["type"]) if t == HUB]
    for a in hubs:
        for b in hubs:
            if a < b and rel[a, b] > 2:
                pa, pb = papers[inst["ctx"]["slot"][a]], papers[inst["ctx"]["slot"][b]]
                edges.append(_edge(f"h{pa}", f"h{pb}", REL_NAMES[rel[a, b]], S_NB, width=3, dashes=[2, 6]))
    cid = inst["ctx"]["comp"]
    for (i, j), lab in zip(inst["probes"], inst["probe_labels"]):
        ci, cj = int(cid[i]), int(cid[j])
        if ci >= 0 and cj >= 0:
            name = ROLE_RELS[lab] if lab < len(ROLE_RELS) else "no edge"
            edges.append(_edge(f"k{ci}", f"k{cj}", f"probe: {name}?", S_PROBE, color="#ff7f00",
                               dashes=[8, 4], width=3, arrows=""))
    a = inst["audit"]
    stages = ["the anchor paper (its KUs and schema edges)",
              f"neighbourhood: {len(papers)} papers (hub links = citation / same field / same subfield)",
              f"mask the target {COMP_TYPES[gi.comp_type[anchor]]}: text removed, role edges cut (type kept)",
              f"hide what paraphrases it: {len(hidden)} KU(s) (claim evidence/implications, method<->procedure)",
              f"hide {len(inst['probes'])} probe pairs from the encoder: the inverse head must recover their "
              f"relation | audit: max visible similarity to target {a['max_ctx_sim']:.2f}"
              f"{' -> FLAGGED (paraphrase leak)' if a['flagged'] else ''}"]
    _write(path, "One training instance", nodes, edges, stages,
           "Step through how a training instance is built. Red star = masked target; grey = hidden from the "
           "encoder; orange dashed = probe pairs whose relation the inverse head must predict.",
           autoplay_ms=2200, stage_mode=True)


def corpus_html(gi, path, n_papers=150, ku_level=False, seed=0, split=None):
    """Papers appear one by one; edges = citations, proxy grounded links, shared subfield."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(gi.papers(split)) if split else np.arange(gi.n_papers)
    linked = [p for p in pool if gi.paper_nbrs.get(int(p))]
    picks = list(dict.fromkeys(linked[:n_papers // 2] + list(rng.choice(pool, min(n_papers, len(pool)), replace=False))))
    picks = [int(p) for p in picks][:n_papers]
    # group by subfield so shared-subfield structure is visible
    picks.sort(key=lambda p: (gi.paper_field[p], gi.paper_subfield[p]))
    order = {p: k for k, p in enumerate(picks)}
    fields = sorted({gi.paper_field[p] for p in picks})
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"]
    fcol = {f: pal[i % len(pal)] for i, f in enumerate(fields)}
    nodes, edges = [], []
    for p in picks:
        n = _hub_node(gi, p, f"h{p}", order[p] + 1)
        n.update(color=fcol[gi.paper_field[p]], label=gi.field_names[gi.paper_field[p]][:16], shape="dot",
                 size=10 + 2 * min(len(gi.paper_nbrs.get(p, {})), 10), font={"size": 10})
        nodes.append(n)
        if ku_level:
            for c in gi.paper_comps[p]:
                c = int(c)
                nodes.append(_ku_node(gi, c, f"k{c}", order[p] + 1, size=5, label=""))
                edges.append(_edge(f"h{p}", f"k{c}", "member", order[p] + 1, width=0.5))
    for p in picks:
        for q, w in gi.paper_nbrs.get(p, {}).items():
            if q in order and p < q:
                edges.append(_edge(f"h{p}", f"h{q}", "cites" if w == 1 else "proxy",
                                   max(order[p], order[q]) + 1, width=3))
    by_sub = {}
    for p in picks:
        by_sub.setdefault(int(gi.paper_subfield[p]), []).append(p)
    for ps in by_sub.values():
        for a, b in zip(ps, ps[1:]):                    # a chain, not a clique: readable
            edges.append(_edge(f"h{a}", f"h{b}", "same_subfield", max(order[a], order[b]) + 1, width=1))
    stages = ["(empty)"] + [f"+ paper {k + 1}/{len(picks)}: {gi.field_names[gi.paper_field[p]]}"
                            for k, p in enumerate(picks)]
    legend_fields = {gi.field_names[f]: fcol[f] for f in fields}
    _write(path, "Corpus graph" + (" (KU level)" if ku_level else ""), nodes, edges, stages,
           f"{len(picks)} papers (colour = field, size = number of linked papers). Black = citation, "
           f"orange = proxy grounded link (near-duplicate methods), grey = same subfield. "
           f"Corpus totals: {gi.n_papers} papers, {gi.n_comps} KUs, {int((gi.pp_w == 1).sum())} citations, "
           f"{int((gi.pp_w == 2).sum())} proxy links.",
           autoplay_ms=120, legend_override=legend_fields)


# ---------------------------------------------------------------- page template
def _write(path, title, nodes, edges, stages, caption, autoplay_ms=500, stage_mode=False, legend_override=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    legend = legend_override or {t: c for t, c in TYPE_COLOR.items()
                                 if any(n.get("group") == t for n in nodes)}
    page = _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__CAPTION__", caption) \
        .replace("__DATA__", json.dumps({"nodes": nodes, "edges": edges, "stages": stages,
                                         "autoplay": autoplay_ms, "stageMode": stage_mode,
                                         "legend": legend}, default=str))
    with open(path, "w") as f:
        f.write(page)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="vendor/vis-network.min.js"></script>
<script>if(!window.vis){document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"><\/script>')}</script>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--text:#1a1a1a;--muted:#666;--line:#ddd;--accent:#e7298a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141414;--panel:#1e1e1e;--text:#eee;--muted:#aaa;--line:#333}}
:root[data-theme="dark"]{--bg:#141414;--panel:#1e1e1e;--text:#eee;--muted:#aaa;--line:#333}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 -apple-system,system-ui,sans-serif;background:var(--bg);color:var(--text)}
header{padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:17px;margin:0 0 4px}.cap{color:var(--muted);font-size:13px;margin:0}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
button{font:inherit;padding:5px 11px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--text);cursor:pointer}
button:hover{border-color:var(--accent)}#stage{font-weight:600;min-width:220px}
#net{height:calc(100vh - 150px);min-height:420px;background:var(--panel)}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
input[type=range]{width:160px}
</style></head><body>
<header><h1>__TITLE__</h1><p class="cap">__CAPTION__</p></header>
<div class="bar">
 <button id="play">▶ Play</button><button id="prev">◀</button><button id="next">▶|</button><button id="reset">⟲ Reset</button>
 <input id="slider" type="range" min="0" value="0"><span id="stage"></span>
 <button id="phys">Physics: on</button><button id="fit">Fit</button>
 <div class="legend" id="legend"></div>
</div>
<div id="net"></div>
<script>
const D = __DATA__;
const last = D.stages.length - 1;
let k = 0, timer = null, physics = true;
const nodes = new vis.DataSet(), edges = new vis.DataSet();
const net = new vis.Network(document.getElementById('net'), {nodes, edges}, {
  physics:{solver:'forceAtlas2Based', forceAtlas2Based:{gravitationalConstant:-45, springLength:90}, stabilization:false},
  interaction:{hover:true, tooltipDelay:80}, edges:{smooth:{type:'continuous'}}, nodes:{font:{size:12}}});
document.getElementById('legend').innerHTML = Object.entries(D.legend).map(([t,c])=>`<span><i style="background:${c}"></i>${t}</span>`).join('');
const slider = document.getElementById('slider'); slider.max = last;
function styled(item){
  let s = Object.assign({}, item); delete s.style_from; delete s.show_from;
  for (const [at, st] of (item.style_from||[])) if (k >= at) Object.assign(s, st);
  if (s.opacity !== undefined) { s.color = {background:s.color, border:s.color, highlight:s.color}; }
  return s;
}
function render(){
  const wantN = D.nodes.filter(n => n.show_from <= k).map(styled);
  const ids = new Set(wantN.map(n=>n.id));
  const wantE = D.edges.filter(e => e.show_from <= k && ids.has(e.from) && ids.has(e.to)).map((e,i)=>Object.assign(styled(e),{id:e.from+'>'+e.to+'#'+(e.rel||i)}));
  nodes.remove(nodes.getIds().filter(i => !ids.has(i))); nodes.update(wantN);
  const eids = new Set(wantE.map(e=>e.id)); edges.remove(edges.getIds().filter(i=>!eids.has(i))); edges.update(wantE);
  slider.value = k; document.getElementById('stage').textContent = `${k}/${last}  ${D.stages[k]}`;
}
function step(d){ k = Math.max(0, Math.min(last, k + d)); render(); if (k >= last) stop(); }
function play(){ if (timer) return stop(); if (k >= last) k = 0; timer = setInterval(()=>step(1), D.autoplay); document.getElementById('play').textContent='❚❚ Pause'; }
function stop(){ clearInterval(timer); timer = null; document.getElementById('play').textContent='▶ Play'; }
document.getElementById('play').onclick = play;
document.getElementById('next').onclick = ()=>{stop(); step(1)};
document.getElementById('prev').onclick = ()=>{stop(); step(-1)};
document.getElementById('reset').onclick = ()=>{stop(); k=0; render()};
document.getElementById('fit').onclick = ()=>net.fit({animation:true});
document.getElementById('phys').onclick = ()=>{physics=!physics; net.setOptions({physics:{enabled:physics}}); document.getElementById('phys').textContent='Physics: '+(physics?'on':'off')};
slider.oninput = ()=>{stop(); k = +slider.value; render()};
render(); setTimeout(play, 600);
</script></body></html>
"""
