#!/bin/bash
#SBATCH --job-name=paper_a
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#
# SCI-JEPA Paper A on a Slurm cluster (set up for GWDG Grete; site settings in site.env next to this file).
# Run on the LOGIN NODE, in a VS Code terminal (Remote-SSH), from the repository root:
#
#   bash cluster/grete/run_paper_a.sh setup            # once: conda env (VENV), packages, models, dashboard assets (needs internet)
#   bash cluster/grete/run_paper_a.sh submit check     # environment + real data + GPU + dashboard check (minutes)
#   bash cluster/grete/run_paper_a.sh submit smoke     # synthetic-data test of every component (~25 min)
#   bash cluster/grete/run_paper_a.sh submit           # the full pipeline on one GPU; opens the dashboard in your browser
#   bash cluster/grete/run_paper_a.sh submit dist      # the same runs split over the job's GPUs, then merged
#   bash cluster/grete/run_paper_a.sh tunnel [JOBID]   # reconnect the browser to a running job's dashboard
#   bash cluster/grete/run_paper_a.sh follow <chain file>   # keep it connected through a whole job chain
#   bash cluster/grete/run_paper_a.sh serve            # a short job that only serves the saved dashboard
#
# Configuration: the Parameters cell of notebooks/paper_a_local.ipynb, as environment variables
#   N_PAPERS SIZE EPOCHS PATIENCE SEEDS EXPS SPEED LLM_MODEL LLM_ON_MAC LLM_QUERIES FIELD_BALANCE FIELD_ALPHA
#   SHARDS_PER_GPU (dist: training processes per GPU) BATCH_WORKERS (batch-building processes per training process)
#   OPEN_BROWSER DATASET (arxiv | pubmed) SPLIT (a stored split folder)     (see: sci-jepa run --help)
# e.g.  N_PAPERS=5000 SIZE=S SEEDS="0 1 2 3 4" EXPS="BENCH E10" bash cluster/grete/run_paper_a.sh submit dist
# Job size: PARTITION GRES TIME CPUS MEM, e.g. PARTITION=kisski GRES=gpu:A100:4 CPUS=64 ... submit dist
#
# Browser: submit / tunnel forward the job's dashboard to localhost:8765 (or the next free port) on the
# login node; VS Code forwards it to your computer and opens it (settings: docs/cluster_grete.md).
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # inside a job: passed from submit
export REPO
SITE_ENV="${SITE_ENV:-${REPO}/cluster/grete/site.env}"
# shellcheck source=site.env
source "${SITE_ENV}" || { echo "cannot read ${SITE_ENV}"; exit 1; }
PY="${VENV}/bin/python"
CMD="${1:-run}"                      # setup | submit | tunnel | follow | watch | serve | run | dist | smoke | check | build | audit

export DATASET="${DATASET:-arxiv}"
[[ -n "${SPLIT:-}" ]] && export PAPER_A_SPLIT="${SPLIT}"   # a stored split (sci-jepa split): no re-sampling / re-splitting
export PAPER_A_PARQUET="${PAPER_A_PARQUET:-${DATA_ROOT}/normalized/${DATASET}}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LLM_MODELS_SP="$(printf "%s" "${LLM_MODELS:-}" | tr "," " ")"   # LLM_MODELS may be comma- or space-separated
MODELS="${MODELS:-Alibaba-NLP/gte-modernbert-base malteos/scincl BAAI/bge-base-en-v1.5 BAAI/bge-m3 \
sentence-transformers/all-MiniLM-L6-v2 sshleifer/tiny-gpt2 ${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct} \
${LLM_MODELS_SP:-Qwen/Qwen3-8B allenai/Olmo-3-7B-Instruct ibm-granite/granite-3.3-8b-instruct microsoft/Phi-4-mini-instruct}}"
mkdir -p "${PAPER_A_INDEX_DIR}" "${PAPER_A_OUT}" "${LOGS}"

# ---------------------------------------------------------------- login node: setup
if [[ "${CMD}" == "setup" ]]; then
  echo "== ${VENV} =="
  if [[ ! -x "${PY}" ]]; then                # create the conda env (plain venv if conda is missing)
    CONDA="${CONDA_EXE:-$(dirname "${PYTHON_BASE}")/conda}"
    if [[ -x "${CONDA}" ]]; then "${CONDA}" create -y -p "${VENV}" -c conda-forge --override-channels python=3.10 pip || exit 1
    else "${PYTHON_BASE}" -m venv "${VENV}" || exit 1; fi
  fi
  "${VENV}/bin/pip" install -q --upgrade pip
  "${VENV}/bin/pip" install -q -r "${REPO}/requirements.txt" || { echo "pip install failed"; exit 1; }
  "${VENV}/bin/pip" install -q -e "${REPO}" "huggingface_hub[hf_xet]" || { echo "pip install -e failed"; exit 1; }
  for M in ${MODELS}; do                       # compute nodes have no internet: download every model now
    echo "== ${M} -> ${HF_HOME}"
    "${VENV}/bin/hf" download "${M}" --exclude "*.pth" --exclude "*.onnx" --exclude "onnx/*" --exclude "openvino/*" >/dev/null \
      || echo "WARNING: ${M} not downloaded"
  done
  "${VENV}/bin/sci-jepa" fetch-assets          # the dashboard's JS libraries, into the package
  "${PY}" -c "import torch, sci_jepa; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| sci_jepa', sci_jepa.__version__)"
  echo "setup done. Next: bash cluster/grete/run_paper_a.sh submit check"
  exit 0
fi

# ---------------------------------------------------------------- login node: tunnel to a job's dashboard
tunnel() {    # $1 = job id
  local JOB="$1" NODE="" PORT="" LPORT=8765 STATE
  echo "waiting for job ${JOB} to start ..."
  while :; do
    STATE=$(squeue -h -j "${JOB}" -o %T 2>/dev/null)
    [[ -z "${STATE}" ]] && { echo "job ${JOB} is not in the queue any more (log: ${LOGS}/paper_a_${JOB}.out)"; return 1; }
    [[ "${STATE}" == RUNNING ]] && break
    sleep 10
  done
  NODE=$(squeue -h -j "${JOB}" -o %N)
  echo "job ${JOB} runs on ${NODE}; waiting for its dashboard (it starts before the KU graph build) ..."
  # the job's own results folder, from its log (" output : ..."), so PAPER_A_OUT need not be set here
  local OUT="${PAPER_A_OUT}" SJ="" O2
  until [[ -n "${SJ}" && -f "${SJ}" ]]; do
    [[ -n "$(squeue -h -j "${JOB}" 2>/dev/null)" ]] \
      || { echo "job ended before the dashboard started:"; tail -20 "${LOGS}/paper_a_${JOB}.out"; return 1; }
    O2=$(sed -n 's/^ output : //p' "${LOGS}/paper_a_${JOB}.out" 2>/dev/null | head -1)
    [[ -n "${O2}" ]] && OUT="${O2}"
    SJ="${OUT}/dashboard/server_${JOB}.json"
    [[ -f "${SJ}" ]] || sleep 5
  done
  PORT=$(grep -o '"port": [0-9]*' "${SJ}" | grep -o '[0-9]*$')
  # login nodes are shared: skip ports someone else listens on (ss by full path: tmux sessions may lack /usr/sbin)
  local SS; SS="$(command -v ss || echo /usr/sbin/ss)"
  free_port() { while "${SS}" -ltn 2>/dev/null | grep -q "127.0.0.1:${LPORT} "; do LPORT=$((LPORT + 1)); done; }
  free_port
  echo "================================================================"
  echo " dashboard: ${NODE}:${PORT}  ->  http://localhost:${LPORT}/"
  echo " VS Code opens it in your browser (Ports tab: 'Paper A dashboard')."
  echo " Ctrl-C closes the tunnel only; the job keeps running. Reconnect: bash cluster/grete/run_paper_a.sh tunnel ${JOB}"
  echo "================================================================"
  while [[ -n "$(squeue -h -j "${JOB}" 2>/dev/null)" ]]; do   # re-opened if the connection drops; ends with the job
    local OLD="${LPORT}"; free_port
    [[ "${LPORT}" != "${OLD}" ]] && echo " port ${OLD} is taken now: dashboard -> http://localhost:${LPORT}/"
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new \
        -L "127.0.0.1:${LPORT}:localhost:${PORT}" "${NODE}"
    sleep 5
  done
  echo "job ${JOB} finished. Results: ${OUT}   Log: ${LOGS}/paper_a_${JOB}.out"
}

if [[ "${CMD}" == "tunnel" ]]; then
  JOB="${2:-$(squeue -h -u "${USER}" -n paper_a -o %i | sort -n | tail -1)}"
  [[ -z "${JOB}" ]] && { echo "no paper_a job in the queue; give a job id"; exit 1; }
  tunnel "${JOB}"; exit $?
fi

# ---------------------------------------------------------------- login node: follow a chain (submit_chain.sh)
# Tunnels to whichever job of the chain is running, one after the other, until no job of the chain is left.
# Without a chain file: the newest chain. To survive a closed laptop, use `watch` (below) instead.
if [[ "${CMD}" == "follow" ]]; then
  CHAIN="${2:-$(ls -t "${LOGS}"/chain_*.txt 2>/dev/null | head -1)}"
  [[ -f "${CHAIN}" ]] || { echo "usage: run_paper_a.sh follow <chain file (printed by submit_chain.sh)>"; exit 1; }
  echo "following chain ${CHAIN} (jobs: $(tr '\n' ' ' < "${CHAIN}"))"
  last=""
  while :; do
    mine=$(squeue -h -u "${USER}" -o "%i %T")                  # all of my jobs: never fails on finished ids
    left=$(grep -Fxf <(awk '{print $1}' <<< "${mine}") "${CHAIN}")
    if [[ -z "${left}" ]]; then
      # FOLLOW_CHAINS (run_60k.sh): a list of chain files; a newer one (the next phase) is followed too
      nxt=""; for _ in $(seq 20); do
        nxt=$(tail -1 "${FOLLOW_CHAINS:-/dev/null}" 2>/dev/null)
        [[ -n "${nxt}" && -f "${nxt}" && "${nxt}" != "${CHAIN}" ]] && break
        nxt=""; [[ -n "${FOLLOW_CHAINS:-}" ]] || break; sleep 30
      done
      [[ -z "${nxt}" ]] && { echo "chain finished $(date +%T)"; exit 0; }
      echo "== $(date +%T) next phase: following chain ${nxt}"; CHAIN="${nxt}"; last=""; continue
    fi
    run=$(awk '$2 == "RUNNING" {print $1}' <<< "${mine}" | grep -Fxf "${CHAIN}" | head -1)
    if [[ -n "${run}" && "${run}" != "${last}" ]]; then
      echo "== $(date +%T) following job ${run}"; last="${run}"
      tunnel "${run}"                                          # returns when that job ends
    fi
    sleep 15
  done
fi

# ---------------------------------------------------------------- login node: watch (follow that survives sleep)
# `follow` inside a tmux session on this login node: it keeps the tunnel to the chain's running job while your
# laptop sleeps or VS Code reconnects. Open http://localhost:8765/ again after waking (VS Code Ports tab).
if [[ "${CMD}" == "watch" ]]; then
  CHAIN="${2:-$(ls -t "${LOGS}"/chain_*.txt 2>/dev/null | head -1)}"
  [[ -f "${CHAIN}" ]] || { echo "no chain file; usage: run_paper_a.sh watch [chain file]"; exit 1; }
  S=sci-jepa-dashboard
  if tmux has-session -t "${S}" 2>/dev/null; then
    echo "already watching on $(hostname -s): tmux attach -t ${S}   (stop: tmux kill-session -t ${S})"; exit 0
  fi
  FLOG="${LOGS}/$(basename "${CHAIN}" .txt)_follow.log"
  tmux new-session -d -s "${S}" "FOLLOW_CHAINS='${FOLLOW_CHAINS:-}' bash '${REPO}/cluster/grete/run_paper_a.sh' follow '${CHAIN}' 2>&1 | tee -a '${FLOG}'"
  echo "================================================================"
  echo " watching chain $(basename "${CHAIN}") in tmux session '${S}' on login node $(hostname -s)"
  echo " dashboard   : http://localhost:8765/ (VS Code forwards it; Ports tab, label 'Paper A dashboard')"
  echo " VS Code must be connected to the SAME login node ($(hostname -s)) for the forward to work"
  echo " see it      : tmux attach -t ${S}    (detach: Ctrl-b d)    log: ${FLOG}"
  echo " stop it     : tmux kill-session -t ${S}   (the jobs keep running)"
  echo "================================================================"
  exit 0
fi

# ---------------------------------------------------------------- login node: submit
if [[ "${CMD}" == "submit" || "${CMD}" == "serve" ]]; then
  JOBCMD="${2:-run}"; [[ "${CMD}" == "serve" ]] && JOBCMD=serve
  [[ -x "${PY}" ]] || { echo "run 'bash cluster/grete/run_paper_a.sh setup' first"; exit 1; }
  case "${JOBCMD}" in
    check|smoke|serve) DEF_GRES="${GRES_SMALL}"; DEF_TIME=01:00:00 ;;
    split)             DEF_GRES="${GRES_SMALL}"; DEF_TIME=06:00:00 ;;
    dist)              DEF_GRES="${GRES_DIST}";  DEF_TIME=12:00:00 ;;
    *)                 DEF_GRES="${GRES_RUN}";   DEF_TIME=12:00:00 ;;
  esac
  OPTS=(--account="${ACCOUNT}" --partition="${PARTITION:-${DEFAULT_PARTITION}}" --gres="${GRES:-${DEF_GRES}}"
        --time="${TIME:-${DEF_TIME}}" --chdir="${REPO}" --export=ALL --output="${LOGS}/paper_a_%j.out")
  [[ "${JOBCMD}" == dist || -n "${CPUS:-}" ]] && OPTS+=(--cpus-per-task="${CPUS:-32}")
  [[ "${JOBCMD}" == dist || -n "${MEM:-}" ]] && OPTS+=(--mem="${MEM:-200G}")
  JOB=$(sbatch --parsable "${OPTS[@]}" "${REPO}/cluster/grete/run_paper_a.sh" "${JOBCMD}") || exit 1
  echo "submitted job ${JOB} (${JOBCMD}); log: ${LOGS}/paper_a_${JOB}.out"
  # hosted dashboard: compute nodes without internet cannot push, so one pusher runs here until the jobs end
  if [[ -n "${SCIJEPA_PUSH_URL:-}" && -n "${SCIJEPA_PUSH_TOKEN:-}" && "${COMPUTE_HAS_INTERNET}" != 1 ]]; then
    DASH="${PAPER_A_OUT}/dashboard"; mkdir -p "${DASH}"
    if pgrep -u "${USER}" -f "push_dashboard.py ${DASH} " >/dev/null; then
      echo "hosted dashboard: pusher already running for ${DASH}"
    else
      nohup python3 "${REPO}/scripts/push_dashboard.py" "${DASH}" --until-idle --squeue-args "-A ${ACCOUNT}" \
        >> "${LOGS}/push_dashboard.log" 2>&1 &
      echo "hosted dashboard: pushing ${DASH} to ${SCIJEPA_PUSH_URL} until your jobs end (log: ${LOGS}/push_dashboard.log)"
    fi
  fi
  [[ "${JOBCMD}" == smoke || "${JOBCMD}" == split ]] && { echo "no dashboard; follow the log: tail -f ${LOGS}/paper_a_${JOB}.out"; exit 0; }
  case "${OPEN_BROWSER:-1}" in 0|false|False|no) echo "OPEN_BROWSER=0: no tunnel. Later: bash cluster/grete/run_paper_a.sh tunnel ${JOB}"; exit 0 ;; esac
  tunnel "${JOB}"; exit $?
fi

# ---------------------------------------------------------------- inside the GPU job
[[ "${COMPUTE_HAS_INTERNET}" == 1 ]] || export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SCIJEPA_AUTOPUSH=0   # login node pushes
# login nodes often set OMP_NUM_THREADS=1, and --export=ALL carries it into the job: use the job's cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "${REPO}" || exit 1
echo "================================================================"
echo " SCI-JEPA Paper A | cmd=${CMD} | job ${SLURM_JOB_ID:-none} on $(hostname) | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/ gpu    : /'
if [[ -n "${PAPER_A_SPLIT:-}" ]]; then echo " data   : stored split ${PAPER_A_SPLIT}"
else echo " data   : ${PAPER_A_PARQUET} ($(ls "${PAPER_A_PARQUET}"/*.parquet 2>/dev/null | wc -l) shards)"; fi
echo " papers : ${N_PAPERS:-500} | size ${SIZE:-local} | epochs ${EPOCHS:-40} | seeds ${SEEDS:-0 1 2} | exps: ${EXPS-BENCH}"
echo " llm    : ${LLM_MODELS:-${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}}"
echo " resume : ${RESUME:-0} (checkpoint every ${CKPT_EVERY:-2000} steps)${CHAIN_FILE:+ | chain ${CHAIN_FILE}}"
echo " output : ${PAPER_A_OUT}"
echo "================================================================"

RC=0
case "${CMD}" in
  run)
    "${PY}" -m sci_jepa.pipeline || RC=$? ;;
  dist)
    IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0}"; NG=${#DEVS[@]}
    # SHARDS_PER_GPU training processes share each GPU (one run uses a few GB of a 40 GB slice); every shard gets
    # its share of the CPUs, and builds its batches in BATCH_WORKERS processes (default: its CPUs minus one, at
    # most 8: more were not faster in the profile, scaling/README.md)
    N=$(( NG * ${SHARDS_PER_GPU:-1} ))
    THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / N )); (( THREADS < 1 )) && THREADS=1
    W=$(( THREADS > 1 ? THREADS - 1 : 0 )); (( W > 8 )) && W=8
    export BATCH_WORKERS="${BATCH_WORKERS:-${W}}"
    echo " gpus   : ${NG} (${CUDA_VISIBLE_DEVICES:-none}) x ${SHARDS_PER_GPU:-1} shards = ${N} shards, ${THREADS} CPUs and ${BATCH_WORKERS} batch workers each"
    # 1. one dashboard server for the whole job (the shards write into the same folder)
    CUDA_VISIBLE_DEVICES="" "${PY}" -m sci_jepa.pipeline --stage serve > "${LOGS}/paper_a_${SLURM_JOB_ID}_serve.log" 2>&1 &
    SERVER=$!
    # 2. KU graph + graphs + shape audit, once (the shards load the cache)
    CUDA_VISIBLE_DEVICES="${DEVS[0]}" "${PY}" -m sci_jepa.pipeline --stage build || { RC=$?; kill ${SERVER}; echo "build failed"; exit ${RC}; }
    # 3. SHARDS_PER_GPU processes per GPU; each claims the next arm x seed run nobody has taken (quick comparison systems, trained arms, LLMs).
    #    Claims belong to this job: clear those of an earlier job of the chain, so its unfinished runs resume here
    find "${PAPER_A_OUT}" -mindepth 2 -maxdepth 2 -type d -name claims -exec rm -rf {} + 2>/dev/null
    export SCI_JEPA_CLAIM=1
    PIDS=()
    for (( i = 0; i < N; i++ )); do
      D="${DEVS[$(( i % NG ))]}"
      CUDA_VISIBLE_DEVICES="${D}" OMP_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS} SCI_JEPA_SHARD="${i}/${N}" \
        "${PY}" -m sci_jepa.pipeline --stage shard --shard "${i}/${N}" > "${LOGS}/paper_a_${SLURM_JOB_ID}_shard${i}.log" 2>&1 &
      PIDS+=($!); echo " shard ${i}/${N} on ${D}: log ${LOGS}/paper_a_${SLURM_JOB_ID}_shard${i}.log"
    done
    for i in "${!PIDS[@]}"; do
      wait "${PIDS[$i]}" || { RC=1; echo "shard ${i} failed: tail ${LOGS}/paper_a_${SLURM_JOB_ID}_shard${i}.log"; }
    done
    # 4. every run finished: summaries, findings, leaderboards, figures, infrastructure, PDF report. Otherwise
    #    (a job of a chain that ran out of runs before time) the merge only records what is missing
    CUDA_VISIBLE_DEVICES="${DEVS[0]}" "${PY}" -m sci_jepa.pipeline --stage merge || RC=$?
    # a chain (submit_chain.sh) whose work is complete: cancel its remaining jobs, so they do not hold GPUs
    if [[ -n "${CHAIN_FILE:-}" && -f "${CHAIN_FILE}" ]] && grep -q '"complete": true' "${PAPER_A_OUT}/infra/status.json" 2>/dev/null; then
      LATER=$(awk -v me="${SLURM_JOB_ID}" 'f{print} $1==me{f=1}' "${CHAIN_FILE}")
      [[ -n "${LATER}" ]] && { echo "all runs finished: cancelling the rest of the chain: ${LATER}"; scancel ${LATER}; }
      # a run in phases (run_60k.sh all): this phase is complete -> submit the next one as a new chain, once
      if [[ -n "${NEXT_SEEDS:-}" && ! -f "${PAPER_A_OUT}/next_phase_started" ]]; then
        date > "${PAPER_A_OUT}/next_phase_started"
        echo "phase complete: starting the next phase (SEEDS=${NEXT_SEEDS}, ${NEXT_CHAIN_LEN:-14} jobs)"
        NEXT=()
        for k in PAPER_A_OUT SPLIT N_PAPERS SIZE EXPS EPOCHS PATIENCE LLM_QUERIES CKPT_EVERY SPEED LLM_MODELS; do
          [[ -n "${!k:-}" ]] && NEXT+=("${k}=${!k}")
        done
        ( cd "${REPO}" && env -u CHAIN_FILE -u CHAIN_LEFT -u CHAIN_SBATCH bash cluster/grete/submit_chain.sh \
            "${NEXT_CHAIN_LEN:-14}" -- "${NEXT[@]}" "SEEDS=${NEXT_SEEDS}" ) \
          && ls -t "${LOGS}"/chain_*.txt | head -1 >> "${PAPER_A_OUT}/chains.txt" \
          || echo "could not submit the next phase; start it by hand (run_60k.sh phase2)"
      fi
    fi
    echo "dashboard stays up ${KEEP_SERVING:-30} more minutes"; sleep $(( ${KEEP_SERVING:-30} * 60 ))
    kill ${SERVER} ;;
  smoke)
    "${PY}" -m sci_jepa.smoke_test "${@:2}" || RC=$? ;;
  build)
    "${PY}" -m sci_jepa.data.ku_graph --set max_papers="${N_PAPERS:-500}" device=cuda || RC=$? ;;
  audit)
    "${PY}" -m sci_jepa.training.train --audit-shapes --size "${SIZE:-local}" --set max_papers="${N_PAPERS:-500}" device=cuda || RC=$? ;;
  serve)
    "${PY}" - <<'EOF' || RC=$?
import json, os, socket, time
from pathlib import Path
from sci_jepa.dashboard.serve import start_server
DASH = os.path.join(os.environ["PAPER_A_OUT"], "dashboard")
port = int(start_server(DASH, 8765).rstrip("/").rsplit(":", 1)[1])   # first free port from 8765
info = json.dumps({"node": socket.gethostname(), "port": port, "job": os.environ.get("SLURM_JOB_ID")})
for n in ("server.json", f"server_{os.environ.get('SLURM_JOB_ID')}.json"):
    Path(DASH, n).write_text(info)
print("serving", DASH, "on port", port, flush=True)
time.sleep(55 * 60)
EOF
    ;;
  check)
    "${PY}" "${REPO}/scripts/check_environment.py" || RC=$? ;;
  split)                             # SPLIT_ARGS="--out ... --train-papers ..." (sci-jepa split --help)
    "${PY}" -m sci_jepa.data.split ${SPLIT_ARGS:-} || RC=$? ;;
  *)
    echo "unknown command '${CMD}' (setup | submit | tunnel | serve | run | dist | smoke | check | split | build | audit)"; exit 2 ;;
esac
echo "done rc=${RC}; results under ${PAPER_A_OUT}"
exit ${RC}
