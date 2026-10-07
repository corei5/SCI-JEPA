#!/bin/bash
# Graph-input JEPA (train_kg_graph.py) on JURECA: one run per message-passing depth.
#
#   mkdir -p scripts/poc/graph_runs
#   sbatch scripts/poc/graph.sbatch
#
# Step 1 builds the graph embedding cache if it is missing: the Qwen
# Result/Evidence/Claim vectors are reused and only Paper/Method/Implication are
# embedded, on the GPU. Compute nodes have no internet, so the model must already
# be downloaded (from a login node) into $HF_HUB_CACHE.
#
# Step 2 trains --layers 0 1 2 3 in parallel, one per GPU. layers=0 has no message
# passing and is the control. Hyperparameters are the best-MRR text-vector config of
# the grid (scripts/poc/grid/15639747/table.md), so claim R@1/MRR compare directly.
#
#SBATCH --job-name=jepa_graph
#SBATCH --account=westai0096
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=scripts/poc/graph_runs/slurm_%j.out
#SBATCH --error=scripts/poc/graph_runs/slurm_%j.err

set -euo pipefail

PROJECT_DIR=/p/project1/westai0096/research/SCI-JEPA
cd "${PROJECT_DIR}"
source env.sh
export HF_HUB_OFFLINE=1

EMBED=Qwen/Qwen3-Embedding-0.6B
OUT="${PROJECT_DIR}/scripts/poc/graph_runs/${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

# Prepend, never replace: torch itself reaches Python through the module's PYTHONPATH.
# env -u: sentence-transformers hands SENTENCE_TRANSFORMERS_HOME (_st_cache) to transformers
# as its cache, which hides the Qwen download in $HF_HUB_CACHE from an offline load.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/poc:${PYTHONPATH}" env -u SENTENCE_TRANSFORMERS_HOME \
    python - "${EMBED}" <<'EOF'
import sys
from pathlib import Path

import torch
from train_kg import CACHE_DIR, KG_DIR
from train_kg_graph import build_cache

model = sys.argv[1]
slug = model.split("/")[-1].lower()
kg = Path(KG_DIR).name
path = Path(f"{CACHE_DIR}/{kg}_{slug}_graph.pt")
if path.exists():
    print(f"graph cache present: {path}")
else:
    build_cache(KG_DIR, path, Path(f"{CACHE_DIR}/{kg}_{slug}.pt"), model, torch.device("cuda"))
EOF

pids=()
for layers in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${layers} python scripts/poc/train_kg_graph.py \
        --embed-model "${EMBED}" --batch-size 256 --hidden 1024 --dropout 0.125 --lr 1e-4 \
        --layers "${layers}" --epochs 500 --patience 10 --seed 0 \
        --results "${OUT}/results.tsv" --out "${OUT}/ckpt_layers${layers}.pt" \
        --plot "${OUT}/plot_layers${layers}.png" \
        >"${OUT}/layers${layers}.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}" || echo "a run failed; see ${OUT}/*.log" >&2; done

column -t -s $'\t' "${OUT}/results.tsv"
