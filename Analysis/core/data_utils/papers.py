"""
Compatibility shim.

`PapersDataset` (Part 1: one graph per paper) moved to the top-level
`dataset_creation/` package. `core.get_data` still imports it from here.

New code should import from `dataset_creation.papers` directly.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset_creation.papers import *               # noqa: F401,F403,E402
from dataset_creation.papers import (               # noqa: F401,E402
    NODE_TYPE_ID,
    NODE_TYPE_VOCAB,
    PapersDataset,
    parse_paper,
)
