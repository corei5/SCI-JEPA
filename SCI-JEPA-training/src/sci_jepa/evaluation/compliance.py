"""
Regulation and ethics assessment (sci-jepa compliance): SCI-JEPA checked against EU AI ethics, EU data protection and
the other frameworks that govern AI, with evidence computed from the project's own artifacts wherever possible.

  EU AI Act (Regulation (EU) 2024/1689)            risk class, research and open-source exemptions, GPAI threshold,
                                                   transparency, AI literacy, voluntary high-risk practices
  HLEG Ethics Guidelines for Trustworthy AI (2019) the seven requirements, scored like the ALTAI self-assessment (2020)
  GDPR (Regulation (EU) 2016/679)                  lawful basis and research safeguards, data minimisation (columns the
                                                   pipeline reads), personal data in the KU texts (scan), erasure, DPIA
  EU copyright: DSM Directive (EU) 2019/790        text-and-data-mining exceptions (Art. 3 research, Art. 4 opt-out),
                                                   dataset licence and attribution
  Product Liability Directive (EU) 2024/2853       software as a product; free open-source research release
  Council of Europe Framework Convention on AI     human rights, democracy, rule of law (obligations on states)
  OECD AI Principles (2019, rev. 2024), UNESCO Recommendation on the Ethics of AI (2021)   mapped to the checks above
  ISO/IEC 42001:2023, ISO/IEC 23894:2023           AI management system and risk management (organisational)
  European Code of Conduct for Research Integrity (ALLEA, 2023); Living guidelines on the responsible use of
  generative AI in research (European Commission, 2024)   AI in research practice

status: met · ready (prepared, waiting for a person's sign-off) · partly · gap · manual (a human or organisational step;
cannot be checked from files) · n/a · info. `--docs` also writes the documents that close the gaps that files can close:
docs/MODEL_CARD.md, docs/DATASHEET.md (regenerated each run), docs/governance/*.md (drafts with a sign-off block; never
overwritten once they exist), release/NOTICE, release/exclude_papers.txt.
This is an engineering self-assessment, not legal advice; the data-protection officer and legal office decide.
Writes <exp>/compliance.json, <exp>/compliance.md and dashboard/compliance.json.
    sci-jepa compliance <results folder> [--exp BENCH]
"""
import argparse
import glob
import json
import os
import pickle
import re

import numpy as np

PERSONAL_COLS = ("source_authors", "summary_authors", "oa_authorships", "text_raw", "text_sanitized")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
ETAL_RE = re.compile(r"\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+)? et al\.")
GPAI_FLOP = 1e23        # Commission guidelines (July 2025): indicative threshold for a general-purpose AI model
SYSTEMIC_FLOP = 1e25    # AI Act Art. 51(2): presumption of systemic risk
A100_KW, PUE, GRID_KG_PER_KWH = 0.40, 1.2, 0.38     # assumptions for the energy estimate (stated in the report)

FRAMEWORKS = {
    "ai_act": "EU AI Act (Regulation (EU) 2024/1689)",
    "hleg": "EU HLEG Ethics Guidelines for Trustworthy AI (2019) · ALTAI (2020)",
    "gdpr": "GDPR (Regulation (EU) 2016/679)",
    "copyright": "EU copyright: DSM Directive (EU) 2019/790",
    "pld": "Product Liability Directive (EU) 2024/2853",
    "coe": "Council of Europe Framework Convention on AI (CETS No. 225, 2024)",
    "intl": "OECD AI Principles (2019, rev. 2024) · UNESCO Recommendation on the Ethics of AI (2021)",
    "iso": "ISO/IEC 42001:2023 · ISO/IEC 23894:2023",
    "research": "Research integrity: ALLEA Code (2023) · EC living guidelines on generative AI in research (2024)",
}


# ---------------------------------------------------------------- evidence from the project's artifacts
def _load(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def evidence(out_dir, exp="BENCH", texts_path=None, log=print):
    exp_dir = os.path.join(out_dir, exp)
    E = {"safety": _load(os.path.join(exp_dir, "safety.json")) or {},
         "science": _load(os.path.join(exp_dir, "science_safety.json")) or {},
         "infra": _load(os.path.join(exp_dir, "infrastructure.json")) or {},
         "paired": _load(os.path.join(exp_dir, "paired_v2.json")) or {}}
    from sci_jepa.data.ku_graph import READ_COLS
    E["read_cols"] = list(READ_COLS)
    E["personal_cols_read"] = [c for c in READ_COLS if c in PERSONAL_COLS]
    # results: one config, for the split folder and the index
    cfg = None
    for f in sorted(glob.glob(os.path.join(exp_dir, "*_seed0.json"))):
        m = _load(f) or {}
        if m.get("model_dir"):
            cfg = m.get("config"); break
    E["config"] = cfg or {}
    split_dir = (cfg or {}).get("split_dir")
    E["split_manifest"] = bool(split_dir and os.path.exists(os.path.join(split_dir, "manifest.json")))
    readme = os.path.join(os.path.dirname(os.path.dirname(split_dir)), "README.md") if split_dir else None
    lic = None
    if readme and os.path.exists(readme):
        mm = re.search(r"^license:\s*(\S+)", open(readme).read(), re.M)
        lic = mm.group(1) if mm else None
    E["dataset_license"] = lic
    # personal data in the KU texts the model reads
    if texts_path is None and cfg:
        cand = sorted(glob.glob(os.path.join(cfg.get("index_dir", ""), "paper_a_ku_index_*.texts.pkl")),
                      key=os.path.getmtime)
        texts_path = cand[-1] if cand else None
    if texts_path and os.path.exists(texts_path):
        texts = pickle.load(open(texts_path, "rb"))["comp_text"]
        n = len(texts); em = orc = et = 0; ex = []
        for t in texts:
            t = t or ""
            a, b, c = EMAIL_RE.search(t), ORCID_RE.search(t), ETAL_RE.search(t)
            em += bool(a); orc += bool(b); et += bool(c)
            if a and len(ex) < 3:
                ex.append(EMAIL_RE.sub("<email>", t[max(0, a.start() - 60):a.end() + 40]))
        E["pii"] = {"kus": n, "email": em, "orcid": orc, "et_al": et, "examples_masked": ex, "file": os.path.basename(texts_path)}
        log(f"[compliance] scanned {n:,} KU texts: {em} with an e-mail, {orc} with an ORCID, {et:,} citing 'Name et al.'")
        del texts
    # training compute (FLOPs) of the largest arm: 6 x parameters x tokens processed
    comp = (E["infra"].get("compute") or [])
    tok = 189                                            # tokens per instance at size S (8 papers x ~23 KUs + hubs)
    flops = [6.0 * (c.get("params") or 0) * (c.get("steps_mean") or 0) * (c.get("batch_size") or 0) * tok for c in comp]
    E["train_flop_max"] = max(flops) if flops else None
    E["params_max"] = max([c.get("params") or 0 for c in comp] or [0]) or None
    tot = E["infra"].get("total") or {}
    a100 = tot.get("a100_equiv_hours")
    E["energy"] = None if a100 is None else {
        "a100_equiv_hours": a100, "gpu_hours": tot.get("gpu_hours"), "runs": tot.get("runs"),
        "kwh": a100 * A100_KW * PUE, "kg_co2e": a100 * A100_KW * PUE * GRID_KG_PER_KWH,
        "assumptions": f"A100 {A100_KW * 1000:.0f} W at full load, PUE {PUE}, grid {GRID_KG_PER_KWH} kg CO2e/kWh (German mix)"}
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proj = os.path.dirname(root)
    E["docs"] = {k: os.path.exists(os.path.join(proj, p)) for k, p in
                 (("code_documentation", "docs/CODE_DOCUMENTATION.md"), ("model_card", "docs/MODEL_CARD.md"),
                  ("readme", "README.md"))}
    E["docs"]["safety_report"] = os.path.exists(os.path.join(exp_dir, "safety.md"))
    return E


# ---------------------------------------------------------------- governance documents (drafts to sign) and the model card
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GOV = {"dpo": "docs/governance/DPO_PACKAGE.md", "literacy": "docs/governance/AI_LITERACY.md",
       "removal": "docs/governance/REMOVAL_PROCEDURE.md", "licence": "docs/governance/LICENCE_DECISION.md",
       "contact": "release/CONTACT"}
SIGN = "\n## Sign-off\n\nSigned off by: ____________________\nRole: ____________________\nDate: ____________________\n"


def _signed(path):
    """A governance draft counts as signed when its 'Signed off by:' line holds a name (not the blank line)."""
    if not os.path.exists(path):
        return False
    m = re.search(r"^Signed off by:\s*(.+)$", open(path).read(), re.M)
    return bool(m and "___" not in m.group(1) and len(m.group(1).strip()) > 2)


def gov_status(proj=PROJ):
    out = {}
    for k, p in GOV.items():
        path = os.path.join(proj, p)
        ok = os.path.exists(path)
        if k == "contact":                                  # a plain file: filled when it is not the placeholder
            out[k] = {"path": p, "exists": ok, "signed": ok and "to be set" not in open(path).read()}
        else:
            out[k] = {"path": p, "exists": ok, "signed": _signed(path)}
    return out


def _abstain(E, rel):
    """Recommended abstention threshold for P(supports / contradicts) from the reliability bins of the released arm's
    calibration check: the smallest threshold whose answered pairs are right >= 95 % of the time."""
    bins = []
    for tag, r in ((E["safety"].get("models") or {}).items()):
        if tag.startswith(rel + "_seed"):
            bins += ((r.get("calibration") or {}).get("reliability") or [])
    if not bins:
        return None
    best = None
    for t in [x / 100 for x in range(50, 100, 5)]:
        ans = [b for b in bins if b["conf"] >= t]
        n = sum(b["n"] for b in ans)
        if n == 0:
            break
        acc = sum(b["acc"] * b["n"] for b in ans) / n
        cov = n / sum(b["n"] for b in bins)
        if acc >= 0.95:
            best = {"threshold": t, "accuracy": acc, "coverage": cov}
            break
    return best


def _per_domain(exp_dir, rel):
    """Mean over seeds and the three retrieval tasks of the ratio to chance per research domain (released arm)."""
    acc = {}
    for f in sorted(glob.glob(os.path.join(exp_dir, f"{rel}_seed*.json"))):
        T = (_load(f) or {}).get("tasks") or {}
        for k in ("forward", "inverse", "abduction"):
            for d, v in ((T.get(k) or {}).get("by_domain") or {}).items():
                if v.get("ratio") is not None and v.get("n", 0) >= 20:
                    acc.setdefault(d, []).append((v["ratio"], v["n"]))
    return {d: {"ratio": float(np.mean([r for r, _ in v])), "n": int(np.mean([n for _, n in v]))} for d, v in acc.items()}


def governance_docs(E, out_dir, exp="BENCH", rel="jepa+rgat", proj=PROJ, log=print):
    """Writes the model card and datasheet (regenerated) and the governance drafts (only when missing)."""
    exp_dir = os.path.join(out_dir, exp)
    os.makedirs(os.path.join(proj, "docs", "governance"), exist_ok=True)
    os.makedirs(os.path.join(proj, "release"), exist_ok=True)
    sci = (E["science"].get("data") or {})
    S = E["safety"].get("systems") or {}
    card = (S.get(rel) or {}).get("card") or {}
    v = lambda k: (card.get(k) or {}).get("v")
    pct = lambda x: "–" if x is None else f"{100 * x:.0f} %"
    dom = _per_domain(exp_dir, rel)
    ab = _abstain(E, rel)
    en, pii = E.get("energy") or {}, E.get("pii") or {}
    contact_p = os.path.join(proj, GOV["contact"])
    if not os.path.exists(contact_p):
        open(contact_p, "w").write("Contact for questions, problems and removal requests: (to be set: an issue tracker "
                                   "URL and a functional e-mail address of the project)\n")
    contact = open(contact_p).read().strip()
    fails = sorted(k for k, f in ((S.get(rel) or {}).get("flags") or {}).items() if f == "fail")   # the released arm only
    mc = [f"# Model card: SCI-JEPA ({rel})", "",
          "A joint-embedding predictive (JEPA) world model over the knowledge units (KUs) of research papers, with a "
          "relation-aware graph encoder (R-GAT). It predicts the representation of a hidden KU (e.g. a result) from the "
          "visible KUs and ranks candidate texts; it judges whether evidence supports or contradicts a claim. "
          "**It generates no text.**", "",
          "## Intended use", "- Ranking candidate results, methods or claims for a paper, as an aid to a human reader.",
          "- Judging claim-evidence pairs, with human review.", "- Research on scientific world models.", "",
          "## Not permitted", "- Decisions about people, papers, grants, admissions, hiring or promotion (this would make "
          "it a high-risk system under the EU AI Act, Annex III 3-4).", "- Fact-checking, medical, legal or safety "
          "decisions without an expert.", "- Use outside the training distribution (English arXiv papers, published up to "
          "2024) without checking quality first. **Language: English only.**", "",
          "## Human oversight", "- `consistency(..., abstain_below=t)` returns `decision = \"abstain\"` when the larger "
          "probability is below t. " + (f"Recommended t = {ab['threshold']:.2f}: answered pairs are right "
                                        f"{100 * ab['accuracy']:.0f} % of the time, {100 * ab['coverage']:.0f} % are answered."
                                        if ab else "Recommended t: pending the calibration check."),
          "- **Do not filter rankings by their score**: the most confident retrieval answers are not more accurate "
          f"(selective-prediction gain {pct(v('selective_gain'))}).", "",
          "## Training data", "- laion/Scientific-Summaries (CC-BY-4.0; LAION e.V. 2025, arXiv:2502.19413), arXiv subset, "
          "60,000 / 5,000 / 5,000 papers, time-based split (test papers 2025-2026).", "- KUs come from LLM-written "
          "summaries" + (f"; {pct((sci.get('faithfulness') or {}).get('ku_supported'))} of checked KUs are fully supported "
                         f"by their paper (Bespoke-MiniCheck-7B)" if (sci.get("faithfulness") or {}).get("n_kus") else "") + ".",
          "- No author, affiliation or e-mail field is read; contact data inside KU texts is redacted in releases.", "",
          "## Performance by research domain (ratio of balanced MRR to chance, mean of 3 tasks and the seeds)", "",
          "| domain | × chance | queries per run |", "|---|---|---|"]
    mc += [f"| {d} | {r['ratio']:.1f} | {r['n']} |" for d, r in sorted(dom.items(), key=lambda kv: -kv[1]["ratio"])]
    mc += ["", "Domains with fewer than 20 test queries are not shown. Full results with 95 % intervals: the project "
           "dashboard and safety report.", "", "## Known limitations (safety checks that fail)"]
    names = {"robustness_ci": "meaning-reversed copies of the true result often rank above it",
             "edit_error": "a reversed copy beats the truth in many questions",
             "identity": "representations identify the source paper (shortcut)",
             "structure": "the typed graph structure barely changes the answers",
             "context_use": "neighbouring papers barely change the prediction",
             "injection_asr": "a wrong answer planted in the paper's own text can win",
             "halluc_on_attempt": "the similarity rule accepts many wrong answers",
             "selective_gain": "confidence does not identify correct answers",
             "false_accept": "a wrong candidate also passes the similarity rule",
             "robust_worst": "the strongest kind of meaning-reversing edit fools it",
             "sycophancy": "contradicting evidence is often judged as supporting",
             "fab_number": "a result with a real paper's numbers swapped in can win",
             "fab_splice": "a result spliced from two papers can win", "overclaim": "overclaimed wording can win",
             "label_gap": "supports and contradicts are recognised unequally",
             "label_gap_ci": "supports and contradicts are recognised unequally (with interval)",
             "robustness": "meaning-reversing edits are ranked near chance",
             "meaning_sensitive": "a reworded truth does not reliably beat a meaning-reversed edit",
             "injection_effect": "planted wrong answers win more often than unrelated planted text"}
    mc += [f"- {names.get(k, k)} ({k})" for k in fails] or ["- none"]
    mc += ["", "Several of these are evaluation-wide properties (e.g. robustness near chance for every system); "
           "values with intervals are in the safety report."]
    mc += ["", "## Environmental impact", (f"- Training of the experiment: {en.get('a100_equiv_hours', 0):.0f} A100-equivalent "
           f"hours, about {en.get('kwh', 0):.0f} kWh and {en.get('kg_co2e', 0):.0f} kg CO2e ({en.get('assumptions')})."
           if en else "- not recorded"), "", "## Licence and attribution",
           "- See LICENSE and NOTICE of the release (decision: docs/governance/LICENCE_DECISION.md).",
           "- Cite the dataset: LAION e.V. (2025), Project Alexandria, arXiv:2502.19413.", "",
           "## Contact, problems and removal requests", f"- {contact}",
           "- Removal procedure: docs/governance/REMOVAL_PROCEDURE.md.", ""]
    open(os.path.join(proj, "docs", "MODEL_CARD.md"), "w").write("\n".join(mc))
    du, rt = sci.get("dual_use") or {}, sci.get("retractions") or {}
    if not rt and (E.get("config") or {}).get("split_dir"):     # science job not run yet: from the paper metadata
        from sci_jepa.evaluation.science_safety import paper_meta
        meta = paper_meta(E["config"]["split_dir"])
        rt = {sp: {"retracted": [p for p, m in meta.items() if m["split"] == sp and m["retracted"]],
                   "withdrawn": [p for p, m in meta.items() if m["split"] == sp and m["withdrawn"]]}
              for sp in ("train", "val", "test")}
    ds = ["# Datasheet: SCI-JEPA training data (split arxiv_60k)", "",
          "- Source: laion/Scientific-Summaries (CC-BY-4.0), arXiv subset; LLM-written structured summaries "
          "(priv-gemini-2.0-flash-lite) split into knowledge units with typed relations and citation links.",
          "- Split: time-based, domain-balanced, 60,000 / 5,000 / 5,000 papers; stored with a manifest.",
          "- Known bias: English only; predominantly Western institutions (dataset card).",
          f"- Personal data: no author / affiliation / e-mail fields read; KU texts with an e-mail address: "
          f"{pii.get('email', '–')}, with an ORCID iD: {pii.get('orcid', '–')} (redacted on release).",
          "- KU faithfulness: " + (f"{pct((sci.get('faithfulness') or {}).get('ku_supported'))} of "
                                    f"{(sci.get('faithfulness') or {}).get('n_kus')} KUs fully supported by the paper text."
                                    if (sci.get("faithfulness") or {}).get("n_kus") else "pending."),
          "- Dual-use share of training KUs (WMDP-near + hazard term): " + (", ".join(
              f"{c} {100 * ((du.get('splits') or {}).get('train', {}).get(c) or {}).get('share', 0):.2f} %"
              for c in du.get("categories", [])) if du else "pending."),
          "- Retracted / withdrawn papers: " + (", ".join(
              f"{sp} {len((rt.get(sp) or {}).get('retracted', []))} / {len((rt.get(sp) or {}).get('withdrawn', []))}"
              for sp in ("train", "val", "test")) + " (listed in release/exclude_papers.txt; dropped from the next "
                                                       "training run)" if rt else "pending."),
          "- Uses: research under the text-and-data-mining exception for research (DSM Art. 3, UrhG § 60d). Full "
          "texts are never redistributed.", ""]
    open(os.path.join(proj, "docs", "DATASHEET.md"), "w").write("\n".join(ds))
    if rt:
        ids = [(p, sp, k) for sp in ("train", "val", "test") for k in ("retracted", "withdrawn") for p in (rt.get(sp) or {}).get(k, [])]
        open(os.path.join(proj, "release", "exclude_papers.txt"), "w").write(
            "# papers dropped from releases and from the next training run (retracted / withdrawn / removal requests)\n"
            + "".join(f"{p}  # {k} ({sp})\n" for p, sp, k in ids))
    notice = os.path.join(proj, "release", "NOTICE")
    open(notice, "w").write(
        "SCI-JEPA\nCopyright the SCI-JEPA authors, TIB - Leibniz Information Centre for Science and Technology.\n\n"
        "Trained on laion/Scientific-Summaries (CC-BY-4.0): LAION e.V. (2025), Project Alexandria: Structured Summaries of "
        "Scientific Papers at Scale, arXiv:2502.19413. The summaries were derived from the papers; the papers' full "
        "texts are not included.\n\nComponents: Alibaba-NLP/gte-modernbert-base (Apache-2.0) text encoder.\n")
    drafts = {
        "dpo": ["# Data-protection package for the TIB data-protection officer", "",
                "Prepared from the project's artifacts (sci-jepa compliance). Please check, amend and sign.", "",
                "## 1. Lawful basis (GDPR Art. 6, Art. 89; BDSG § 27 / Lower Saxony data-protection law)",
                "- Purpose: scientific research on machine learning over published research papers.",
                "- Proposed basis: Art. 6(1)(e) (task in the public interest of a public research institution), with "
                "Art. 89(1) safeguards; alternatively Art. 6(1)(f).",
                "- Safeguards: data minimisation (no author / affiliation / e-mail fields read), pseudonymous paper ids, "
                "no profiling or decisions about people, project storage with group-restricted access, contact data "
                "redacted in any release, removal procedure.", "",
                "## 2. Record of processing (Art. 30), draft entry",
                "- Controller: TIB - Leibniz Information Centre for Science and Technology.",
                "- Data subjects: authors of published papers (indirectly: names cited in summaries, contact data that "
                "appear inside paper texts).",
                "- Categories of data: published scholarly texts; incidental names and contact data; no special categories.",
                "- Recipients: project team; published model weights contain no personal data.",
                "- Transfers to third countries: none (all processing on GWDG, Göttingen).",
                "- Retention: copies of the corpus deleted at the end of the project (UrhG § 60d(5)); models and metrics kept.",
                "- Security (Art. 32): GWDG project storage, access restricted to the project group.", "",
                "## 3. Information duty (Art. 14(5)(b))",
                "- Informing every author is impossible or disproportionate; a project privacy note on the release page "
                "describes the processing and the removal contact.", "",
                "## 4. DPIA screening (Art. 35; EDPB criteria)",
                "- Evaluation / scoring of people: no · automated decisions with legal effect: no · systematic monitoring: "
                "no · sensitive data: no · large scale: published texts, incidental personal data only · matching of "
                "datasets: no · vulnerable subjects: no · innovative technology: yes (ML) · preventing exercise of "
                "rights: no.", "- Result: one criterion of nine; a full DPIA is not required (to be confirmed).", SIGN],
        "literacy": ["# AI literacy record (EU AI Act Art. 4)", "",
                     "Everyone who develops, operates or uses SCI-JEPA in the project has read:",
                     "- docs/MODEL_CARD.md (intended use, limits, human oversight)",
                     "- the safety report (dashboard, Safety tab) and the regulation assessment",
                     "- docs/CODE_DOCUMENTATION.md (for developers)", "",
                     "| name | role | date | material |", "|---|---|---|---|", "| | | | |", SIGN],
        "removal": ["# Removal procedure (GDPR Art. 17, 21; authors' requests)", "",
                    "1. Requests reach the contact in release/CONTACT.",
                    "2. Add the paper id to release/exclude_papers.txt with the reason and date.",
                    "3. Releases: `sci-jepa release export ...` drops excluded papers from any KU text file at once.",
                    "4. Training data: excluded papers are dropped when the split / index is rebuilt for the next training "
                    "run; published weights are replaced at the next release (weights contain no personal data).",
                    "5. Answer the requester within one month (Art. 12(3)).", SIGN],
        "licence": ["# Licence decision for the release", "",
                    "Proposal (to confirm):",
                    "- Code: Apache-2.0 (permissive, patent grant; compatible with all components).",
                    "- Model weights: Apache-2.0, with an acceptable-use clause in the model card (no decisions about "
                    "people; Annex III uses excluded).",
                    "- KU texts (if released): CC-BY-4.0, as the source summaries, with attribution in NOTICE.",
                    "- Not shipped: Bespoke-MiniCheck-7B (CC-BY-NC-4.0; used only for an evaluation check).",
                    "- After signing: put the official Apache-2.0 text in release/LICENSE.", SIGN],
    }
    for k, lines in drafts.items():
        p = os.path.join(proj, GOV[k])
        if not os.path.exists(p):
            open(p, "w").write("\n".join(lines))
    log(f"[compliance] documents: docs/MODEL_CARD.md, docs/DATASHEET.md, docs/governance/ (drafts), release/")
    return {"per_domain": dom, "abstain": ab}


# ---------------------------------------------------------------- the assessment
def _flags(E, keys, tags=None):
    """Worst flag over the given safety checks across the JEPA models, or only the given arms (the model to release)."""
    S = (E["safety"].get("systems") or {})
    if tags:
        S = {t: r for t, r in S.items() if t in tags}
    order = {"fail": 3, "warn": 2, "pass": 1}
    worst, seen = None, []
    for tag, r in S.items():
        if r.get("family") == "comparison":
            continue
        for k in keys:
            f = (r.get("flags") or {}).get(k)
            if f in order:
                seen.append(f)
                if worst is None or order[f] > order[worst]:
                    worst = f
    return worst, len(seen)


def _metric(E, tag, key):
    c = (((E["safety"].get("systems") or {}).get(tag) or {}).get("card") or {}).get(key)
    return None if not c else c.get("v")


def assess(E):
    R = []
    add = lambda fw, ref, req, applies, status, ev, action="": R.append(
        {"framework": fw, "ref": ref, "requirement": req, "applies": applies, "status": status, "evidence": ev,
         "action": action})
    usage = (E["safety"].get("usage") or [])
    flop, par = E.get("train_flop_max"), E.get("params_max")
    sci = (E["science"].get("data") or {})
    rel = "jepa+rgat"                                    # the model planned for release (publication plan)
    G = E.get("gov") or gov_status()
    sig = lambda k, missing="manual": "met" if G[k]["signed"] else ("ready" if G[k]["exists"] else missing)
    rel_tool = E.get("release_tool", False)

    # ---- EU AI Act
    add("ai_act", "Art. 5", "No prohibited practice (manipulation, social scoring, biometric categorisation, ...)", "yes",
        "met", "The system ranks scientific knowledge units; it processes no biometric or behavioural data about people "
        "and makes no decisions about people.")
    add("ai_act", "Art. 6 · Annex III", "Risk class: not high-risk", "yes", "met",
        "Intended use (model card): ranking findings and judging evidence as an aid to a human reader. None of the "
        "Annex III areas applies. It WOULD become high-risk if used to evaluate people, e.g. admissions or assessment "
        "in education, recruitment or promotion of researchers (Annex III 3-4).",
        "Keep the usage restriction 'not for decisions about papers, grants or people' in the model card and licence.")
    add("ai_act", "Art. 2(6), 2(8)", "Research exemption", "yes", "met",
        "Developed and tested solely for scientific research (TIB / L3S); not placed on the market or put into service.")
    add("ai_act", "Art. 2(12)", "Free and open-source release", "on release", sig("licence", "partly"),
        "A release under a free and open-source licence is outside the Act unless it is high-risk, prohibited or an "
        "Art. 50 system; none applies. Licence decision: " + ("signed." if G["licence"]["signed"] else
        f"drafted in {G['licence']['path']} (Apache-2.0 for code and weights, CC-BY-4.0 for KU texts)."),
        "" if G["licence"]["signed"] else "Confirm and sign the licence decision; put the official text in release/LICENSE.")
    if flop:
        add("ai_act", "Art. 3(63), 51-55", "General-purpose AI model obligations", "no", "n/a",
            f"Not a GPAI model: a task-specific, non-generative model with {par / 1e6:.1f} M parameters, trained with about "
            f"{flop:.1e} FLOPs (largest arm), far below the indicative GPAI threshold ({GPAI_FLOP:.0e}) and the "
            f"systemic-risk presumption ({SYSTEMIC_FLOP:.0e}).")
    add("ai_act", "Art. 50", "Transparency for AI interacting with people / generating content", "no", "n/a",
        "Does not interact with people as a chatbot and generates no text, image, audio or video.",
        "Still state in any interface that rankings come from an AI model (good practice).")
    add("ai_act", "Art. 4", "AI literacy of staff who develop and use the system", "yes", sig("literacy"),
        "Applies to providers and deployers since 2 Feb 2025, also in research institutions. Record: "
        + ("signed." if G["literacy"]["signed"] else f"template in {G['literacy']['path']} (model card, safety report, docs)."),
        "" if G["literacy"]["signed"] else "Each team member reads the listed material and is entered in the record; sign.")
    w, n = _flags(E, ["robustness_ci", "edit_error", "injection_asr", "paraphrase_stable"], [rel])
    add("ai_act", "Art. 15 (voluntary, Art. 95)", "Accuracy, robustness and cybersecurity, declared",
        "voluntary", {"fail": "gap", "warn": "partly", "pass": "met", None: "manual"}[w],
        f"Measured (safety R2-R3) on {n} checks of {rel}, the model to release: worst flag '{w}'. A wrong answer planted in the paper's "
        f"own KUs wins {100 * (_metric(E, rel, 'injection_asr') or 0):.0f} % for {rel}; meaning-reversed copies beat the "
        f"truth in {100 * (1 - (_metric(E, rel, 'robustness_ci') or 0)):.0f} % of comparisons.",
        "Declared in docs/MODEL_CARD.md (known limitations). Fixing them needs v2 training with planted distractors "
        "and meaning-reversed negatives (SGWM plan)." if E["docs"].get("model_card") else
        "Declare these limits in the model card; train with planted distractors before a v2.")
    add("ai_act", "Art. 10 (voluntary)", "Data governance: relevant, representative, examined for bias",
        "voluntary", "met" if (sci.get("faithfulness") or {}).get("n_kus") and E.get("exclusions_ready") else "partly",
        "Stored, domain-balanced split with manifest (counts at every step); dataset biases documented by the dataset "
        "card (English, Western institutions). KU faithfulness check: "
        + ("done" if (sci.get("faithfulness") or {}).get("n_kus") else "queued (science-safety job)") + ".",
        ("" if (sci.get("faithfulness") or {}).get("n_kus") else "Finish the KU faithfulness check (science-safety job). ")
        + ("Retracted / withdrawn papers are listed in release/exclude_papers.txt and dropped at the next training run."
           if E.get("exclusions_ready") else "Remove retracted papers from training."))

    # ---- HLEG / ALTAI: seven requirements
    sel = _metric(E, rel, "selective_gain")
    ab = E.get("abstain")
    add("hleg", "1", "Human agency and oversight", "yes",
        "met" if ab and E.get("abstain_api") and E["docs"].get("model_card") else ("partly" if sel is not None and sel < 0 else "met"),
        "Designed as an aid: it ranks real texts and does not decide. But its confidence is not usable for oversight: "
        f"the most confident 40 % of answers are {abs(100 * sel):.0f} points LESS accurate ({rel})." if sel is not None
        else "Designed as an aid that ranks real texts; confidence check pending.",
        (f"Done: consistency(..., abstain_below={ab['threshold']:.2f}) abstains on uncertain claim-evidence pairs "
         f"(answered pairs {100 * ab['accuracy']:.0f} % right, {100 * ab['coverage']:.0f} % answered); the model card "
         "tells users not to filter rankings by their score." if ab and E.get("abstain_api") else
         "Add a calibrated abstention option; tell users not to filter by confidence."))
    w, n = _flags(E, ["robustness_ci", "robust_worst", "edit_error", "injection_asr", "injection_effect", "ece",
                      "sycophancy", "fab_number", "fab_splice", "overclaim"], [rel])
    add("hleg", "2", "Technical robustness and safety", "yes", {"fail": "gap", "warn": "partly", "pass": "met", None: "manual"}[w],
        f"Safety risk card R2-R4 for {rel}, worst flag '{w}' over {n} checks: fooled by planted answers and meaning-reversed "
        "copies; calibration of supports / contradicts is good.", "Needs v2 training: planted distractors and "
        "meaning-reversed copies as training negatives (SGWM plan); limits are declared in the model card.")
    pii = E.get("pii") or {}
    add("hleg", "3", "Privacy and data governance", "yes", "met" if not E["personal_cols_read"] else "partly",
        f"The pipeline reads no author, affiliation or full-text column (columns read: {len(E['read_cols'])}); "
        + (f"{pii.get('email', 0)} of {pii.get('kus', 0):,} KU texts contain an e-mail address." if pii else "KU scan not run."),
        ("Release export redacts e-mails / ORCIDs (sci-jepa release); the index is rebuilt redacted for v2."
         if rel_tool else "Mask e-mail addresses in KU texts before release.") if pii.get("email") else "")
    add("hleg", "4", "Transparency: traceability, explainability, communication", "yes",
        "partly" if not E["docs"].get("model_card") else "met",
        "Traceable: stored split with manifest, every run's full config and per-query records, infrastructure log, "
        "code documentation" + (", safety report" if E["docs"].get("safety_report") else "") + ". Explainability: "
        "relation decoder, neuron and world-model views. Model card: " + ("present." if E["docs"].get("model_card") else "missing."),
        "" if E["docs"].get("model_card") else "Write docs/MODEL_CARD.md (intended use, limits from the risk card, data, licence).")
    w, n = _flags(E, ["domain_gap_ci", "high_stakes_gap", "temporal_gap", "field_size_gap", "novelty_penalty",
                      "citation_bias", "hazard_uplift"], [rel])
    add("hleg", "5", "Diversity, non-discrimination and fairness", "yes",
        {"fail": "gap", "warn": "partly", "pass": "met", None: "manual"}[w],
        f"Risk card R5-R6 for {rel}, worst flag '{w}' over {n} checks (domain, high-stakes domains, year, subfield size, "
        f"novelty, popularity): quality even across research domains and years; the corpus "
        "is English-only and Western-skewed (dataset card)." + (f" Per-domain scores ({len(E.get('per_domain') or {})} "
        "domains) and the language limit are in the model card." if E.get("per_domain") else ""),
        "" if E.get("per_domain") else "Report per-domain scores in the paper; state the language limit.")
    en = E.get("energy") or {}
    hz = (sci.get("dual_use") or {}).get("splits", {}).get("train", {})
    add("hleg", "6", "Societal and environmental well-being", "yes", "met" if en else "manual",
        (f"Compute logged: {en.get('gpu_hours', 0):.0f} GPU-slice hours = {en.get('a100_equiv_hours', 0):.0f} A100-hours, "
         f"about {en.get('kwh', 0):.0f} kWh and {en.get('kg_co2e', 0):.0f} kg CO2e ({en.get('assumptions')}; "
         f"{en.get('runs')} runs logged so far). " if en else "")
        + ("Dual-use share of training KUs: " + ", ".join(f"{c} {100 * (hz.get(c) or {}).get('share', 0):.2f} %"
                                                          for c in ("bio", "chem", "cyber")) if hz else
           "Dual-use scan queued (science-safety job)."),
        "Report energy and CO2 in the paper (ML CO2 reporting).")
    add("hleg", "7", "Accountability: auditability, minimising and reporting negative impact, redress", "yes",
        "met" if G["contact"]["signed"] and G["removal"]["exists"] else ("ready" if G["contact"]["exists"] else "partly"),
        "Internal audit of the evaluation done (25 Oct) with fixes (paired benchmark v2); pre-registered hypotheses; "
        "results reproducible from stored configs. No external audit yet; no channel for reporting problems.",
        ("Contact channel: " + ("set." if G["contact"]["signed"] else "placeholder in release/CONTACT: enter an issue "
         "tracker and a functional address.") + " Recommended external check: expert annotation of ~200 items "
         "(sci-jepa validity export)."))

    # ---- GDPR
    add("gdpr", "Art. 5(1)(c)", "Data minimisation", "yes", "met" if not E["personal_cols_read"] else "gap",
        "Columns read by the KU pipeline: " + ", ".join(E["read_cols"][:6]) + ", ... (" + str(len(E["read_cols"])) +
        " in total); none of " + ", ".join(PERSONAL_COLS) + ".")
    add("gdpr", "Art. 4(1), 9", "Personal data in what the model learns from", "yes",
        "partly" if pii.get("email") or pii.get("orcid") else "met",
        (f"Scan of {pii.get('kus', 0):,} KU texts: {pii.get('email', 0)} e-mail addresses, {pii.get('orcid', 0)} ORCID iDs, "
         f"{pii.get('et_al', 0):,} citations of the form 'Name et al.' (names of cited authors: public, scholarly). "
         "No special-category data is processed about identifiable people (papers report aggregate findings).") if pii else "Scan not run.",
        (("Releases: redacted on export and verified (sci-jepa release check). " if rel_tool else "")
         + "Index: rebuild with redaction for the next training run (v2; the running 60k index stays as it is).")
        if pii.get("email") or pii.get("orcid") else "")
    add("gdpr", "Art. 6(1)(e)/(f), 89", "Lawful basis and research safeguards", "yes", sig("dpo"),
        "Scientific research on published papers; safeguards in place: minimisation (no author fields), project storage "
        "with access control, no profiling of people.", "" if G["dpo"]["signed"] else f"Send {G['dpo']['path']} (section 1) to the TIB data-protection officer for sign-off.")
    add("gdpr", "Art. 14(5)(b)", "Information to data subjects (data not obtained from them)", "yes", "met",
        "Exemption for scientific research where informing every author is impossible or disproportionate; the "
        "project page and paper describe the processing.")
    add("gdpr", "Art. 30", "Record of processing activities", "yes", sig("dpo"),
        "Organisational record; draft entry in " + G["dpo"]["path"] + " (section 2).",
        "" if G["dpo"]["signed"] else "The data-protection officer enters it in TIB's record of processing.")
    add("gdpr", "Art. 35", "Data protection impact assessment", "conditional", sig("dpo"),
        "Not likely required: no systematic monitoring, no large-scale special-category data, no decisions about people.",
        "" if G["dpo"]["signed"] else f"Screening against the EDPB criteria drafted ({G['dpo']['path']}, section 4); "
        "the data-protection officer confirms.")
    add("gdpr", "Art. 17", "Right to erasure", "yes", "met" if G["removal"]["exists"] and rel_tool else "partly",
        "The model holds no author data; a paper can be removed from the stored split and the index rebuilt; weights "
        "change only after retraining." + (f" Procedure: {G['removal']['path']}; exclusions applied by the release "
        "export." if G["removal"]["exists"] else ""), "" if G["removal"]["exists"] else "Document the removal procedure in the release.")
    add("gdpr", "Art. 32", "Security of processing", "yes", "met",
        "Data and models on GWDG project storage (group-restricted permissions); no personal data is exported.")

    # ---- copyright / TDM
    lic = E.get("dataset_license")
    add("copyright", "DSM Art. 3", "Text and data mining for scientific research by a research organisation", "yes", "met",
        "TIB is a research organisation; mining the corpus for research is covered, including keeping copies with "
        "appropriate security.")
    notice = os.path.exists(os.path.join(PROJ, "release", "NOTICE"))
    add("copyright", "Dataset licence", "Licence of the training data respected", "yes",
        ("met" if notice and rel_tool else "partly") if lic else "manual",
        f"laion/Scientific-Summaries: {lic or 'unknown'} (summaries; attribution required). Full texts and their licences come "
        "from the sources (arXiv: per-paper licences, often the arXiv non-exclusive distribution licence, which does not "
        "allow redistribution)." + (" Attribution in release/NOTICE; the release export never includes full texts and "
        "`sci-jepa release check` verifies it." if notice and rel_tool else ""),
        "" if notice and rel_tool else "Before release: attribute the dataset; do not redistribute full texts; release "
        "KU texts only where the summary licence allows (CC-BY-4.0).")
    add("copyright", "DSM Art. 4 · AI Act Art. 53(1)(c)", "Opt-outs and copyright policy (TDM outside research / GPAI)", "no",
        "n/a", "Art. 4 opt-outs bind mining outside the Art. 3 research exception, and Art. 53(1)(c) binds GPAI providers; "
        "neither applies to this research model. Good practice: honour machine-readable opt-outs if the data is ever "
        "extended from the web, and before any commercial use.")

    # ---- other frameworks
    add("pld", "Art. 2(2)", "Liability for defective products (software)", "conditional", "info",
        "Free and open-source software developed or supplied outside a commercial activity is excluded; a commercial "
        "product built on the model would be covered (from Dec 2026).", "Keep the release non-commercial or assess with legal.")
    add("coe", "Arts. 1-16", "Human rights, democracy and the rule of law across the AI lifecycle", "info", "info",
        "Obligations fall on the states that ratify; the principles (transparency, accountability, equality, privacy, "
        "reliability) are covered by the HLEG rows above.")
    add("intl", "OECD 1.1-1.5 · UNESCO", "Inclusive growth, human rights, transparency, robustness, accountability", "info",
        "info", "Mapped: 1.1 well-being -> HLEG 6; 1.2 human rights and fairness -> HLEG 5, GDPR; 1.3 transparency -> "
        "HLEG 4; 1.4 robustness and safety -> HLEG 2, safety R2-R4; 1.5 accountability -> HLEG 7. UNESCO adds "
        "environmental sustainability (HLEG 6) and data governance (GDPR rows).")
    add("iso", "ISO/IEC 42001 · 23894", "AI management system and risk management", "voluntary", "info",
        "The risk card (R1-R8), the audit and this assessment are the technical inputs a 42001 system would ask for; "
        "the management system itself is organisational.", "Optional for a research project.")
    add("research", "ALLEA 2023 · EC 2024", "Responsible AI in research: disclose use, stay accountable, protect data",
        "yes", "met", "The paper reports every model, prompt and setting (report section S6); authors remain responsible "
        "for conclusions; no confidential or personal data is sent to external AI services (all models run locally on GWDG).")
    return R


def summary(R, E):
    cnt = {}
    for r in R:
        cnt.setdefault(r["framework"], {}).setdefault(r["status"], 0)
        cnt[r["framework"]][r["status"]] += 1
    altai = {r["ref"]: {"met": 4, "ready": 3, "partly": 2, "gap": 1, "manual": 0}.get(r["status"], 0) for r in R if r["framework"] == "hleg"}
    actions = [{"framework": FRAMEWORKS[r["framework"]], "ref": r["ref"], "action": r["action"], "status": r["status"]}
               for r in R if r["action"] and r["status"] in ("gap", "partly", "manual", "ready")]
    path = {"sign": [a for a in actions if a["status"] == "ready"],
            "v2": [a for a in actions if "v2" in a["action"] and a["status"] != "ready"],
            "people": [a for a in actions if a["status"] in ("manual",) or ("expert" in a["action"] and a["status"] != "ready")]}
    path["other"] = [a for a in actions if a not in path["sign"] + path["v2"] + path["people"]]
    return {"counts": cnt, "altai_scores": altai,
            "classification": "Research system, not placed on the market: AI Act research exemption (Art. 2(6)). As a "
                              "release: minimal-risk AI system, not a general-purpose AI model, no transparency "
                              "obligation; voluntary codes (Art. 95). Becomes high-risk only if used to evaluate people.",
            "actions": actions, "path": path}


def write(out_dir, exp="BENCH", log=print, docs=True):
    E = evidence(out_dir, exp, log=log)
    if docs:
        E.update(governance_docs(E, out_dir, exp, log=log))
        E["docs"]["model_card"] = os.path.exists(os.path.join(PROJ, "docs", "MODEL_CARD.md"))
    E["gov"] = gov_status()
    E["exclusions_ready"] = os.path.exists(os.path.join(PROJ, "release", "exclude_papers.txt"))
    try:
        from sci_jepa.release import check, redact
        E["release_tool"] = redact("mail a.b@c.org, 0000-0002-1825-0097") == "mail <email>, <orcid>" and callable(check)
    except ImportError:
        E["release_tool"] = False
    E["abstain_api"] = "abstain_below" in open(os.path.join(PROJ, "src", "sci_jepa", "inference.py")).read()
    R = assess(E)
    rep = {"frameworks": FRAMEWORKS, "requirements": R, "summary": summary(R, E),
           "evidence": {k: E[k] for k in ("read_cols", "personal_cols_read", "dataset_license", "pii", "train_flop_max",
                                          "params_max", "energy", "docs", "split_manifest", "gov", "abstain", "per_domain",
                                          "release_tool", "exclusions_ready") if k in E},
           "note": "Engineering self-assessment from the project's artifacts, not legal advice; status 'manual' needs a "
                   "human or organisational step, 'ready' a signature on a prepared draft. Dates and thresholds as adopted; check for later amendments."}
    exp_dir = os.path.join(out_dir, exp)
    json.dump(rep, open(os.path.join(exp_dir, "compliance.json"), "w"), indent=1, default=str)
    md = ["# Regulation and ethics assessment", "", rep["summary"]["classification"], "", rep["note"], ""]
    for fw, name in FRAMEWORKS.items():
        rows = [r for r in R if r["framework"] == fw]
        if not rows:
            continue
        md += [f"## {name}", "", "| ref | requirement | applies | status | evidence | action |", "|---|---|---|---|---|---|"]
        md += [f"| {r['ref']} | {r['requirement']} | {r['applies']} | {r['status']} | {r['evidence']} | {r['action']} |"
               for r in rows]
        md.append("")
    open(os.path.join(exp_dir, "compliance.md"), "w").write("\n".join(md) + "\n")
    dash = os.path.join(out_dir, "dashboard")
    if os.path.isdir(dash):
        json.dump(rep, open(os.path.join(dash, "compliance.json"), "w"), default=str)
        import shutil                                     # the documents, readable from the dashboard
        os.makedirs(os.path.join(dash, "governance"), exist_ok=True)
        for p in ["docs/MODEL_CARD.md", "docs/DATASHEET.md", "release/NOTICE", "release/exclude_papers.txt"] + \
                 [v for k, v in GOV.items()]:
            if os.path.exists(os.path.join(PROJ, p)):
                shutil.copy(os.path.join(PROJ, p), os.path.join(dash, "governance", os.path.basename(p)))
    log(f"[compliance] {len(R)} requirements -> {exp_dir}/compliance.json, compliance.md")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", help="results folder (PAPER_A_OUT), e.g. $STORE/SCI-JEPA/arxiv60k_S")
    p.add_argument("--exp", default="BENCH")
    A = p.parse_args(argv)
    write(A.out, A.exp)


if __name__ == "__main__":
    main()
