"""
Compatibility shim.

The knowledge-graph construction code moved OUT of the modelling package to the
top-level `dataset_creation/` package (repo root). Training scripts still do
`from core.data_utils.paper_graph import build_hetero_graph, ASPECTS`, so this
module re-exports it from its new home.

New code should import from `dataset_creation.paper_graph` directly.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset_creation.paper_graph import *          # noqa: F401,F403,E402
from dataset_creation.paper_graph import (          # noqa: F401,E402
    ASPECTS,
    CLAIM_TEXT_FIELDS,
    SUMMARY_FIELD_TO_TYPE,
    build_hetero_graph,
)
