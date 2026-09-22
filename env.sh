#!/usr/bin/env bash
# Activate the SCI-JEPA environment on JURECA (JSC, Stages/2026).
#
#   source env.sh
#
# Why this file exists
# --------------------
# The `.venv` committed upstream was built on macOS/arm64 and cannot run here.
# On JURECA we do NOT pip-install torch: the Stages modules ship a CUDA build
# tuned for the cluster's GPUs and MPI stack. So the layout is:
#
#   modules  -> torch, numpy, scipy, scikit-learn, matplotlib, networkx, tqdm
#   .venv    -> only what the modules lack (torch_geometric, sentence-transformers,
#               transformers, yacs, einops, huggingface_hub)
#
# The venv is created with --system-site-packages so it can see the module
# packages without duplicating them.

set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. cluster software stack -------------------------------------------
# Stages/2026 + GCCcore are already loaded by the default profile; loading
# Stages again would purge these, so only the leaves are requested here.
module load PyTorch/2.9.1 scikit-learn/1.7.1 matplotlib/3.10.5

# --- 2. project virtualenv ------------------------------------------------
# `unset VIRTUAL_ENV` first: a stale value inherited from the old macOS venv
# makes pip and activate disagree about which environment is live.
unset VIRTUAL_ENV
# shellcheck disable=SC1091
source "${PROJECT_DIR}/.venv/bin/activate"

# --- 3. import precedence -------------------------------------------------
# EasyBuild modules publish their packages through PYTHONPATH, and PYTHONPATH
# is searched BEFORE a venv's own site-packages. Without this line the module's
# older `regex` shadows the newer one that `transformers` requires, and
# `import transformers` fails outright. Putting the venv first means: use the
# venv's copy when it has one, fall back to the module otherwise.
export PYTHONPATH="${PROJECT_DIR}/.venv/lib/python3.13/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

# --- 4. caches ------------------------------------------------------------
# $HOME on JSC is small and quota'd; keep model downloads in the project dir.
# Same paths the Analysis/*.sh drivers already use, so they share one cache.
export HF_HOME="${PROJECT_DIR}/_hf_cache"
export SENTENCE_TRANSFORMERS_HOME="${PROJECT_DIR}/_st_cache"
export MPLCONFIGDIR="${PROJECT_DIR}/_mpl_cache"
mkdir -p "${HF_HOME}" "${SENTENCE_TRANSFORMERS_HOME}" "${MPLCONFIGDIR}"

# Login nodes have no GPU; compute nodes do. Harmless either way.
export TOKENIZERS_PARALLELISM=false

set +u

echo "SCI-JEPA env ready: $(python -V 2>&1), torch $(python -c 'import torch;print(torch.__version__)')"
