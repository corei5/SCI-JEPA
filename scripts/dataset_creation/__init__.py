"""
Knowledge-graph dataset creation for SCI-JEPA.

Everything that turns raw paper JSON into a graph lives here, separately from
the modelling code in `Analysis/`:

    schema.py         record parsing + node/edge wiring   (stdlib only)
    paper_graph.py    build_hetero_graph()  -> HeteroData (Part 2 reasoning KG)
    papers.py         PapersDataset         -> one graph per paper (Part 1)
    build_sample.py   small, embedding-free sample KG for inspection
    visualize.py      renders a sample KG to standalone HTML/SVG

See README.md in this directory for the full walkthrough.
"""

from .schema import (
    ASPECTS,
    EDGE_TYPES,
    NODE_TYPES,
    build_tables,
    coverage_report,
    parse_records,
)

__all__ = [
    "ASPECTS",
    "EDGE_TYPES",
    "NODE_TYPES",
    "build_tables",
    "coverage_report",
    "parse_records",
]
