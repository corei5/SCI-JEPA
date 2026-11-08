"""The index's vectors are saved as memory-mapped .npy files and load back identical (data/graph_index.py)."""
import os

import numpy as np
import torch

from sci_jepa.config import SIZES, PaperAConfig
from sci_jepa.data.graph_index import MMAP_KEYS, get_index, load_index, save_index


def test_saved_vectors_load_memory_mapped_and_identical(tmp_path):
    cfg = PaperAConfig().override(**SIZES["tiny"], synthetic=True, embedder="hash", device="cpu", time_T=2016,
                                  test_open_ended=False, proxy_tau=0.6)
    blob = {k: getattr(get_index(cfg), k) for k in ("comp_x", "mp_x", "comp_type", "comp_paper", "paper_field")}
    old = str(tmp_path / "old.pt")
    torch.save(blob, old, pickle_protocol=4)                     # an index saved before the change still loads
    assert all(isinstance(load_index(old)[k], np.ndarray) for k in MMAP_KEYS)
    new = str(tmp_path / "new.pt")
    save_index(blob, new)
    got = load_index(new)
    for k, v in blob.items():
        np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(v))
    assert all(isinstance(got[k], np.memmap) for k in MMAP_KEYS)
    assert "comp_x" not in torch.load(new, weights_only=False)  # the vectors are only in the .npy files
    assert not [f for f in os.listdir(tmp_path) if ".tmp" in f]
