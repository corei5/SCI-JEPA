"""
Synthetic records in the NORMALIZED parquet schema (same 44-key layout as
Scientific-Summaries-Data/normalized/*), for smoke tests. They go through the
real KU parser, placeholder and null-statement filters, and graph builder.

Text is built so the real pathologies exist under the hash embedder:
  * every KU type has its own template vocabulary (a dominant type centroid),
  * a paper's KUs share a few topic words (weak paper identity),
  * supporting evidence copies its claim's words (the paraphrase leak),
  * procedures restate their method (sibling paraphrase),
  * some papers share method wording (proxy grounded edges),
  * contradicting_evidence is often empty, "None.", or a null statement,
  * optional sections are often "N/A".
"""
import json

import numpy as np

_TEMPLATE = {
    "research_context": "prior work background gap motivation field problem",
    "research_question_hypothesis": "question hypothesis test whether predict alternative",
    "methodological_details": "method approach design model estimate analysis",
    "procedures_architectures": "procedure architecture pipeline implemented steps layers",
    "key_results": "result found measured improvement accuracy percent",
    "interpretation_implications": "interpretation implies suggests mechanism theory",
    "contradictions_limitations": "limitation caveat assumption restricted scope",
    "claim": "claim shows demonstrates establishes",
    "evidence": "evidence figure table observed data",
    "implication": "implication future application consequence",
}
_NULLS = ["", "", "None.", "The paper does not provide any contradicting evidence.",
          "No contradicting evidence is presented."]


def synthetic_records(n_papers=300, n_fields=6, seed=0, topic_words=4):
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(4000)]
    method_pool = [" ".join(rng.choice(vocab, 6)) for _ in range(n_papers // 10 + 1)]
    names = ["Physics", "Computer Science", "Mathematics", "Astrophysics", "Statistics", "Biology",
             "Economics", "Chemistry", "Electrical Engineering", "Philosophy"]   # real labels: domains get exercised
    fields = [names[i % len(names)] + ("" if i < len(names) else f" {i}") for i in range(n_fields)]

    def text(kind, topic, extra=""):
        base = _TEMPLATE[kind].split()
        words = list(rng.choice(base, 12)) + list(rng.choice(topic, topic_words)) + list(rng.choice(vocab, 3))
        return " ".join(words) + (" " + extra if extra else "")

    recs, paper_field = [], []
    for p in range(n_papers):
        topic = list(rng.choice(vocab, 8))
        f = int(rng.integers(n_fields))
        paper_field.append(f)
        method_txt = text("methodological_details", topic, method_pool[p % len(method_pool)])
        rec = {"paper_id": f"{(5 + p % 15):02d}{1 + p % 12:02d}.{p:05d}",
               "subset": "arxiv-new",
               "source_year": str(2005 + p % 15) if p % 7 else None, "oa_year": None,
               "field_subfield": f"{fields[f]} — Sub{f}{p % 3}",
               "type_of_paper": ["Theoretical", "Empirical, Methodological", "Methodological", "Review"][p % 4],
               "openalex_id": f"https://openalex.org/W{1000 + p}",
               "research_context": text("research_context", topic),
               "research_question_hypothesis": text("research_question_hypothesis", topic),
               "methodological_details": method_txt,
               "procedures_architectures": method_txt + " " + text("procedures_architectures", topic),
               "key_results": text("key_results", topic, f"{rng.integers(1, 99)}.{rng.integers(10)}%"),
               "interpretation_implications": text("interpretation_implications", topic),
               "contradictions_limitations": text("contradictions_limitations", topic),
               "key_figures_tables": "N/A" if p % 5 == 0 else "Figure 2 shows " + text("evidence", topic),
               "robustness_ablation_notes": None, "data_code_availability": "No links provided.",
               "ethical_considerations": "Not applicable.",
               "executive_summary": "SHOULD NEVER BECOME A KU " + " ".join(topic),
               "three_takeaways": "SHOULD NEVER BECOME A KU"}
        same_field = [q for q in range(p) if paper_field[q] == f]
        rec["oa_referenced_works"] = json.dumps(
            [f"https://openalex.org/W{1000 + int(q)}" for q in rng.choice(same_field, min(3, len(same_field)),
                                                                          replace=False)]) if same_field else None
        claims = []
        for _ in range(int(rng.integers(3, 7))):
            c = text("claim", topic)
            ce = rng.choice(_NULLS) if rng.random() < 0.65 else text("evidence", topic)
            claims.append({"description": c,
                           "supporting_evidence": c + " " + text("evidence", topic) + " (Figure 3)",
                           "contradicting_evidence": str(ce),
                           "implications": text("implication", topic)})
        rec["claims"] = json.dumps(claims)
        recs.append(rec)
    return recs
