# Datasheet: SCI-JEPA training data (split arxiv_60k)

- Source: laion/Scientific-Summaries (CC-BY-4.0), arXiv subset; LLM-written structured summaries (priv-gemini-2.0-flash-lite) split into knowledge units with typed relations and citation links.
- Split: time-based, domain-balanced, 60,000 / 5,000 / 5,000 papers; stored with a manifest.
- Known bias: English only; predominantly Western institutions (dataset card).
- Personal data: no author / affiliation / e-mail fields read; KU texts with an e-mail address: 142, with an ORCID iD: 37 (redacted on release).
- KU faithfulness: pending.
- Dual-use share of training KUs (WMDP-near + hazard term): pending.
- Retracted / withdrawn papers: train 23 / 8, val 1 / 0, test 0 / 0 (listed in release/exclude_papers.txt; dropped from the next training run)
- Uses: research under the text-and-data-mining exception for research (DSM Art. 3, UrhG § 60d). Full texts are never redistributed.
