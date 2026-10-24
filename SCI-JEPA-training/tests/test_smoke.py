"""The synthetic-data smoke test of every component (minutes; CPU is enough).

    python -m pytest -q -m slow          # or directly: sci-jepa smoke-test
"""
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_smoke():
    r = subprocess.run([sys.executable, "-m", "sci_jepa.smoke_test"], capture_output=True, text=True, timeout=3 * 3600)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "[FAIL]" not in out, out[-4000:]
    assert out.count("[ok]") >= 30, out[-4000:]
