#!/bin/bash
# Grid search over the JEPA claim-prediction prototype on JURECA.
#
#   mkdir -p scripts/poc/grid
#   sbatch scripts/poc/grid.sbatch
#
# 360 configurations: 2 embedding models x 3 batch sizes x 4 hidden sizes
# x 3 dropouts x 5 learning rates. Every run gets --epochs 500 --patience 10,
# so most stop long before 500 and the wall time is dominated by the slow ones.
#
# The node is allocated exclusively (dc-hwai is OverSubscribe=EXCLUSIVE), so all
# 4 GPUs are ours either way. These models are tiny MLPs over cached vectors and
# cannot saturate one GPU, so PER_GPU workers share each device.
#
#SBATCH --job-name=jepa_grid
#SBATCH --account=westai0096
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
# No --cpus-per-task: asking for all 128 is refused ("node configuration is not
# available"), and the partition is exclusive, so the job gets the whole node regardless.
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
#SBATCH --output=scripts/poc/grid/slurm_%j.out
#SBATCH --error=scripts/poc/grid/slurm_%j.err

set -euo pipefail

PROJECT_DIR=/p/project1/westai0096/research/SCI-JEPA
cd "${PROJECT_DIR}"
source env.sh

OUT="${PROJECT_DIR}/scripts/poc/grid/${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}/logs"

EMBEDS=(sentence-transformers/all-MiniLM-L6-v2 Qwen/Qwen3-Embedding-0.6B)
BATCHES=(128 256 512)
HIDDENS=(128 256 512 1024)
DROPOUTS=(0.125 0.25 0.375)
LRS=(1e-3 5e-4 1e-4 5e-5 1e-5)

EPOCHS=500
PATIENCE=10
GPUS=4
PER_GPU=2
WORKERS=$((GPUS * PER_GPU))

# Fail before spawning anything if an embedding cache is missing: the workers would
# otherwise race to build the same file, and a Qwen build needs a login-node download.
for e in "${EMBEDS[@]}"; do
    slug=$(printf '%s' "${e##*/}" | tr '[:upper:]' '[:lower:]')
    f="scripts/poc/cache/kg_materials_chemeng_${slug}.pt"
    [[ -f "${f}" ]] || { echo "missing embedding cache ${f}; build it once before the grid" >&2; exit 1; }
done

CONFIGS=()
for e in "${EMBEDS[@]}"; do
    for b in "${BATCHES[@]}"; do
        for h in "${HIDDENS[@]}"; do
            for d in "${DROPOUTS[@]}"; do
                for l in "${LRS[@]}"; do
                    CONFIGS+=("${e} ${b} ${h} ${d} ${l}")
                done
            done
        done
    done
done
echo "grid: ${#CONFIGS[@]} configurations, ${WORKERS} workers, output ${OUT}"

# Worker w takes configs w, w+WORKERS, w+2*WORKERS, ... so the (slow, large-hidden)
# configs spread across workers instead of piling onto one.
run_worker() {
    local wid=$1
    export CUDA_VISIBLE_DEVICES=$((wid % GPUS))
    local tsv="${OUT}/results_w${wid}.tsv"
    local i e b h d l tag
    for ((i = wid; i < ${#CONFIGS[@]}; i += WORKERS)); do
        read -r e b h d l <<<"${CONFIGS[$i]}"
        tag="${e##*/}_b${b}_h${h}_d${d}_lr${l}"
        if ! python scripts/poc/train_kg.py \
            --embed-model "${e}" --batch-size "${b}" --hidden "${h}" \
            --dropout "${d}" --lr "${l}" \
            --epochs "${EPOCHS}" --patience "${PATIENCE}" \
            --results "${tsv}" --plot '' \
            >"${OUT}/logs/${tag}.log" 2>&1; then
            echo "${tag}" >>"${OUT}/failures.txt"
        fi
    done
}

pids=()
for ((w = 0; w < WORKERS; w++)); do
    run_worker "${w}" &
    pids+=($!)
done
# Never abort the sweep because one worker died; the report says what is missing.
for p in "${pids[@]}"; do wait "${p}" || true; done

python scripts/poc/grid_report.py "${OUT}" | tee "${OUT}/table.md"
echo "done: ${OUT}/table.md"
