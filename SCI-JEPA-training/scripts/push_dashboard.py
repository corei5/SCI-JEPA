#!/usr/bin/env python3
"""Push a dashboard folder to the hosted dashboard. The code is src/sci_jepa/dashboard/push.py (training
runs it in-process); this wrapper keeps the login-node command working with plain python3, no install.
`python3 scripts/push_dashboard.py --help` for the options."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sci_jepa.dashboard.push import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
