"""
Use a trained SCI-JEPA model outside training (prototype / production).

    from sci_jepa.inference import SciJEPA
    m = SciJEPA.from_pretrained("paper_a_results/BENCH/models/motif+world_seed0")
    paper = {"research_context": "...", "research_question_hypothesis": "...",
             "methodological_details": "...", "procedures_architectures": "...",
             "key_results": "...", "claims": [{"description": "...", "supporting_evidence": "...",
             "contradicting_evidence": "", "implications": "..."}], "field_subfield": "Computer Science — ML"}
    m.embed(paper)                                   # one vector per knowledge unit (facet embeddings) + paper
    m.predict(paper, hide="result")                  # predicted representation of the result
    m.rank(paper, hide="result", candidates=[...])   # which candidate text is the most likely result
    m.consistency(claim_text, evidence_text)         # P(supports), P(contradicts)

A paper is given as a structured summary in the training data's format (the fields of the
normalized parquet; `claims` may be a list or a JSON string).

Context, as in evaluation (pass a corpus index to get it):
    from sci_jepa.data.graph_index import get_index
    m = SciJEPA.from_pretrained(folder, index=get_index(cfg))   # or m.attach_index(gi)
  - predict / rank / consistency add the n_subgraphs - 1 corpus papers (train split) most similar
    to the paper's visible parts, as neighbouring papers, like the training instances;
  - rank encodes every candidate INSIDE the query paper (the candidate in the hidden KU's place),
    as the evaluation encodes the true KU and its counterfactual variants;
  - consistency(claim, evidence, paper=...) reads the pair inside its paper (edge hidden).
Without an index the paper is encoded alone; rank(in_context=False) and consistency without
`paper` reproduce the earlier, lighter behaviour (each text encoded on its own).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from sci_jepa.data import ku_schema as S
from sci_jepa.config import PaperAConfig, resolve_device
from sci_jepa.data.graph_index import COMP_TYPES, HUB, REL_SAME_FIELD
from sci_jepa.data.instances import build_view
from sci_jepa.data.instances import collate_views
from sci_jepa.data.ku_schema import LEAK_FAMILY, REL_ID, REL_MEMBER, REL_NONE, REL_SAME_PAPER, TEMPLATES
from sci_jepa.models.model import PaperAModel, to_device

_SUP, _CON = REL_ID["supported_by"] - 3, REL_ID["challenged_by"] - 3


class SciJEPA:
    def __init__(self, model, cfg, device, embedder=None, index=None):
        self.model, self.cfg, self.dev = model.eval(), cfg, device
        self._emb = embedder
        self.gi, self._pv = None, None
        if index is not None:
            self.attach_index(index)

    def attach_index(self, gi):
        """Use a corpus KU index (graph_index.get_index) for neighbouring papers. Its KU features
        must come from the same text encoder the model was trained on."""
        if gi.comp_x.shape[1] != self.cfg.feat_dim:
            raise ValueError(f"index features are {gi.comp_x.shape[1]}-d, the model expects {self.cfg.feat_dim}-d")
        self.gi = gi
        train = np.asarray(gi.papers("train"))
        mu = gi.feat_mean()
        V = np.stack([gi.comp_x[gi.paper_comps[int(p)]].astype(np.float32).mean(0) - mu for p in train])
        self._pv = (train, V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8), mu)
        return self

    # ------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path, weights="model", device="auto", index=None):
        """path: a folder written by train.save_model. weights: "model" (end of training) or
        "best" (best validation epoch). index: optional corpus KU index for neighbouring papers."""
        from safetensors.torch import load_file
        raw = json.load(open(os.path.join(path, "config.json")))
        base = PaperAConfig()
        cfg = base.override(**{k: v for k, v in raw.items() if hasattr(base, k) and k not in ("embedder",)})
        dev = torch.device(resolve_device(device))
        model = PaperAModel(cfg)
        f = os.path.join(path, f"{weights}.safetensors")
        if not os.path.exists(f):
            f = os.path.join(path, "model.safetensors")
        missing, unexpected = model.load_state_dict(load_file(f), strict=False)
        if [k for k in missing if not k.startswith("shapes.")]:
            raise RuntimeError(f"weights do not match the model: missing {missing[:5]}")
        return cls(model.to(dev), cfg, dev, index=index)

    @property
    def text_encoder(self):
        if self._emb is None:                            # the same frozen encoder the model was trained on
            from sci_jepa.data.ku_graph import STEmbedder
            self._emb = STEmbedder(self.cfg.embed_model, str(self.dev), self.cfg.embed_max_tokens,
                                   self.cfg.embed_batch, self.cfg.embed_prefix)
        return self._emb

    # ------------------------------------------------------------ building a paper view
    def _kus(self, paper):
        rec = dict(paper)
        if isinstance(rec.get("claims"), list):
            rec["claims"] = json.dumps(rec["claims"])
        kus, edges, _ = S.record_to_kus(rec, self.cfg.ku_min_chars, self.cfg.optional_kus)
        if not kus:
            raise ValueError("no knowledge units found: give at least a few summary fields")
        return kus, edges, str(rec.get("field_subfield") or "")

    def _view(self, kus, edges, field, hide=None, given=None):
        """Tokens of one paper (hub + KUs). hide: a KU type to mask (added as a mask token if the
        paper does not have one). given: only these KU types stay visible."""
        types = [t for t, _, _ in kus]
        keep = list(range(len(kus)))
        if given is not None:
            keep = [i for i in keep if types[i] in given]
        if hide is not None:
            fam = set(LEAK_FAMILY.get(hide, (hide,)))
            first = next((i for i in range(len(kus)) if types[i] == hide), None)
            keep = [i for i in keep if types[i] not in fam]      # the hidden KU and what restates it
            keep = keep + ([first] if first is not None else [])
        texts = [kus[i][1] for i in keep]
        X = self.text_encoder.encode(texts) if texts else np.zeros((0, self.cfg.feat_dim), np.float32)
        hub = self.text_encoder.encode([field.split("—")[0].strip() or "Unknown"])[0]
        x = np.concatenate([hub[None], X], 0).astype(np.float32)
        typ = np.asarray([HUB] + [COMP_TYPES.index(types[i]) for i in keep], np.int64)
        T = len(typ)
        rel = np.full((T, T), REL_SAME_PAPER, np.int64)
        rel[0, 1:] = rel[1:, 0] = REL_MEMBER
        pos = {i: k + 1 for k, i in enumerate(keep)}
        for a, r, b in edges:
            if a in pos and b in pos:
                rel[pos[a], pos[b]] = rel[pos[b], pos[a]] = REL_ID[r]
        np.fill_diagonal(rel, REL_NONE)
        m = None
        if hide is not None:
            if first is None or first not in pos:                # no such KU: add an empty mask token
                x = np.concatenate([x, np.zeros((1, x.shape[1]), np.float32)]); typ = np.append(typ, COMP_TYPES.index(hide))
                rel = np.pad(rel, ((0, 1), (0, 1)), constant_values=REL_SAME_PAPER); rel[-1, 0] = rel[0, -1] = REL_MEMBER
                rel[-1, -1] = REL_NONE; m = len(typ) - 1
            else:
                m = pos[first]
            x[m] = 0.0
            role = ~np.isin(rel[m], [REL_NONE, REL_SAME_PAPER, REL_MEMBER])
            rel[m, role] = REL_SAME_PAPER; rel[role, m] = REL_SAME_PAPER
        view = {"x": x, "type": typ, "comp": np.arange(len(typ)) - 1, "slot": np.zeros(len(typ), np.int64),
                "deg": np.full(len(typ), len(typ) - 1, np.int64), "rel": rel}
        mask = np.zeros(len(typ), bool)
        if m is not None:
            mask[m] = True
        return view, mask, m, [types[i] for i in keep]

    # ------------------------------------------------------------ context from the corpus
    def _add_neighbours(self, view, mask, field):
        """Append the n_subgraphs - 1 training papers of the corpus index most similar to the paper's
        visible KUs (centred mean text embedding), as neighbouring papers (slots 1..), like a
        training instance. Own tokens stay first, so mask positions do not move."""
        if self.gi is None or self.cfg.n_subgraphs <= 1:
            return view, mask
        gi, cfg = self.gi, self.cfg
        train, V, mu = self._pv
        own = (view["type"] != HUB) & ~mask
        if not own.any():
            return view, mask
        q = view["x"][own].mean(0) - mu
        nb = [int(p) for p in train[np.argsort(-(V @ (q / (np.linalg.norm(q) + 1e-8))))[:cfg.n_subgraphs - 1]]]
        nv = build_view(gi, nb, [gi.paper_comps[p][:cfg.max_comp_per_paper] for p in nb], cfg)
        T0, T1 = len(view["type"]), len(nv["type"])
        rel = np.full((T0 + T1, T0 + T1), REL_NONE, np.int64)
        rel[:T0, :T0], rel[T0:, T0:] = view["rel"], nv["rel"]
        coarse = S.split_field(field)[0]
        for k in np.nonzero(nv["type"] == HUB)[0]:          # own hub <-> neighbour hubs of the same field
            if str(gi.field_names[gi.paper_field[nb[int(nv["slot"][k])]]]) == coarse:
                rel[0, T0 + k] = rel[T0 + k, 0] = REL_SAME_FIELD
        out = {"x": np.concatenate([view["x"], nv["x"]]).astype(np.float32),
               "type": np.concatenate([view["type"], nv["type"]]),
               "comp": np.concatenate([np.full(T0, -1), nv["comp"]]),
               "slot": np.concatenate([view["slot"], nv["slot"] + 1]),
               "deg": np.concatenate([view["deg"], nv["deg"]]), "rel": rel}
        return out, np.concatenate([mask, np.zeros(T1, bool)])

    # ------------------------------------------------------------ uses
    @torch.no_grad()
    def embed(self, paper):
        """Facet embeddings: the context encoder's state of every knowledge unit (with neighbouring
        papers when an index is attached), plus their mean as a paper embedding.
        -> {"kus": [(type, vector)], "paper": vector}"""
        kus, edges, field = self._kus(paper)
        view, mask, _, types = self._view(kus, edges, field)
        n_own = len(view["type"])
        view, mask = self._add_neighbours(view, mask, field)
        v = to_device(collate_views([view], [mask]), self.dev)
        h = self.model.context(v)[0, 1:n_own].float().cpu().numpy()
        return {"kus": list(zip(types, h)), "paper": h.mean(0)}

    @torch.no_grad()
    def predict(self, paper, hide="result", task=None):
        """Predicted representation of the hidden KU type. task: "forward" / "inverse" /
        "abduction" shows only that reasoning template's given parts."""
        kus, edges, field = self._kus(paper)
        given = set(TEMPLATES[task][0]) if task else None
        view, mask, m, _ = self._view(kus, edges, field, hide=hide, given=given)
        view, mask = self._add_neighbours(view, mask, field)
        v = to_device(collate_views([view], [mask]), self.dev)
        h = self.model.context(v)
        return self.model.predictor(h, v, torch.tensor([m], device=self.dev),
                                    torch.tensor([COMP_TYPES.index(hide)], device=self.dev))[0].float().cpu()

    @torch.no_grad()
    def rank(self, paper, hide, candidates, task=None, in_context=True):
        """Rank candidate texts for the hidden KU type. -> [(score, text)] best first.
        in_context: each candidate is encoded inside the paper, in the hidden KU's place (as the
        evaluation encodes the true KU and its counterfactuals); False = each text on its own."""
        pred = self.predict(paper, hide, task)
        X = self.text_encoder.encode(list(candidates))
        if in_context:
            kus, edges, field = self._kus(paper)
            view, _, _, types = self._view(kus, edges, field)
            t = COMP_TYPES.index(hide)
            pos = next((k for k in range(1, len(view["type"])) if view["type"][k] == t), None)
            if pos is None:                              # the paper has no KU of that type: add a slot
                T = len(view["type"])
                rel = np.pad(view["rel"], ((0, 1), (0, 1)), constant_values=REL_SAME_PAPER)
                rel[T, 0] = rel[0, T] = REL_MEMBER; rel[T, T] = REL_NONE
                view = {"x": np.concatenate([view["x"], np.zeros((1, view["x"].shape[1]), np.float32)]),
                        "type": np.append(view["type"], t), "comp": np.append(view["comp"], T - 1),
                        "slot": np.append(view["slot"], 0), "deg": np.full(T + 1, T, np.int64), "rel": rel}
                pos = T
            views = []
            for x in X:
                v_ = {**view, "x": view["x"].copy()}
                v_["x"][pos] = x
                views.append(v_)
            keys = self.model.target(to_device(collate_views(views), self.dev))[:, pos].float().cpu()
        else:
            views = []
            for x in X:
                views.append({"x": np.stack([np.zeros_like(x), x]).astype(np.float32),
                              "type": np.asarray([HUB, COMP_TYPES.index(hide)]), "comp": np.asarray([-1, 0]),
                              "slot": np.zeros(2, np.int64), "deg": np.ones(2, np.int64),
                              "rel": np.asarray([[REL_NONE, REL_MEMBER], [REL_MEMBER, REL_NONE]])})
            keys = self.model.target(to_device(collate_views(views), self.dev))[:, 1].float().cpu()
        s = F.cosine_similarity(pred[None], keys).numpy()
        return sorted(zip(s.tolist(), candidates), reverse=True)

    @torch.no_grad()
    def consistency(self, claim, evidence, paper=None, abstain_below=None):
        """P(supports), P(contradicts) for a claim and a piece of evidence (link hidden), from the
        relation decoder. paper: read the pair inside its paper (and neighbouring papers with an
        index), as in evaluation; claim / evidence are matched to the paper's KUs or added to it.
        abstain_below: human oversight; "decision" is "abstain" when the larger probability is below it
        (these probabilities are calibrated; the recommended value is in the model card). The retrieval
        ranking's confidence is NOT reliable: do not filter rankings by their score."""
        if paper is None:
            X = self.text_encoder.encode([claim, evidence])
            view = {"x": np.concatenate([np.zeros((1, X.shape[1])), X]).astype(np.float32),
                    "type": np.asarray([HUB, COMP_TYPES.index("claim"), COMP_TYPES.index("evidence")]),
                    "comp": np.asarray([-1, 0, 1]), "slot": np.zeros(3, np.int64), "deg": np.full(3, 2),
                    "rel": np.asarray([[REL_NONE, REL_MEMBER, REL_MEMBER], [REL_MEMBER, REL_NONE, REL_SAME_PAPER],
                                       [REL_MEMBER, REL_SAME_PAPER, REL_NONE]])}
            v = to_device(collate_views([view]), self.dev)
            h = self.model.context(v)
            i, j = 1, 2
        else:
            kus, edges, field = self._kus(paper)
            kus = list(kus)
            idx = {}
            for t, txt in (("claim", claim), ("evidence", evidence)):
                k = next((n for n, (tt, x, _) in enumerate(kus) if tt == t and x.strip() == txt.strip()), None)
                if k is None:
                    kus.append((t, txt, {})); k = len(kus) - 1
                idx[t] = k
            edges = [(a, r, b) for a, r, b in edges if {a, b} != {idx["claim"], idx["evidence"]}]  # link hidden
            view, mask, _, _ = self._view(kus, edges, field)
            view, mask = self._add_neighbours(view, mask, field)
            v = to_device(collate_views([view], [mask]), self.dev)
            h = self.model.context(v)
            i, j = idx["claim"] + 1, idx["evidence"] + 1    # +1: the hub is token 0
        logit = self.model.inverse(h[:, i], h[:, j])[0]
        p = torch.softmax(logit[[_SUP, _CON]], -1).cpu().numpy()
        dec = "supports" if p[0] >= p[1] else "contradicts"
        if abstain_below is not None and float(p.max()) < abstain_below:
            dec = "abstain"
        return {"supports": float(p[0]), "contradicts": float(p[1]), "decision": dec}
