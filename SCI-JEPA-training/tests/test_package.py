"""Fast checks of the package itself (seconds, no data, no GPU): python -m pytest -q -m "not slow" """
import importlib
import os
import pkgutil
import subprocess
import sys

import pytest

import sci_jepa

SCRIPTS = {"sci_jepa.pipeline", "sci_jepa.evaluate", "sci_jepa.smoke_test"}   # run on import: not imported here


def all_modules():
    return [m.name for m in pkgutil.walk_packages(sci_jepa.__path__, "sci_jepa.")]


@pytest.mark.parametrize("name", [m for m in all_modules() if m not in SCRIPTS])
def test_module_imports(name):
    importlib.import_module(name)


def test_packaged_files():
    root = os.path.dirname(sci_jepa.__file__)
    for f in ("dashboard/dashboard.html", "models/placeholder_shapes.json"):
        assert os.path.isfile(os.path.join(root, f)), f


def test_cli_lists_every_command():
    from sci_jepa.cli import COMMANDS, usage
    for cmd, (module, _) in COMMANDS.items():
        assert cmd in usage()
        assert importlib.util.find_spec(module) is not None, module


@pytest.mark.parametrize("cmd", ["run", "evaluate", "train", "build-index"])
def test_cli_help(cmd):
    r = subprocess.run([sys.executable, "-m", "sci_jepa.cli", cmd, "--help"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage" in r.stdout.lower()


def test_paths_come_from_the_environment():
    code = "from sci_jepa.config import PaperAConfig as C; c = C(); print(c.parquet_dirs, c.index_dir, c.out_dir)"
    env = {**os.environ, "PAPER_A_PARQUET": "/a,/b", "PAPER_A_INDEX_DIR": "/i", "PAPER_A_OUT": "/o"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.stdout.strip() == "('/a', '/b') /i /o", r.stdout + r.stderr


def test_experiment_presets_are_consistent():
    from sci_jepa.config import EXPERIMENTS, PAPER_EXPERIMENTS
    assert "BENCH" in EXPERIMENTS and "BENCH" in PAPER_EXPERIMENTS
    assert set(PAPER_EXPERIMENTS) <= set(EXPERIMENTS)
