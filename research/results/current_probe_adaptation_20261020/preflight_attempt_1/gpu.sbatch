#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
R=/path/to/phyre-unc/results/current_probe_adaptation_20261020
I=${SLURM_ARRAY_TASK_ID:-0}
export PHYRE_ENCODER_ENDPOINT="$R/encoder$((I%2))/rpc/ENDPOINT.json"
case "$1" in
  preflight) exec .venv-ml/bin/python -u "$R/code/preflight.py" ;;
  pools) exec .venv-ml/bin/python -u "$R/code/build_pools.py" --shard "$I" --nshards 16 ;;
  cache_smoke) exec .venv-ml/bin/python -u "$R/code/collect.py" --smoke ;;
  collect) exec .venv-ml/bin/python -u "$R/code/collect.py" --shard "$I" --nshards 16 ;;
  train_smoke|train)
    arms=(true_control mixed)
    args=()
    if [[ "$1" == train_smoke ]]; then args=(--smoke); fi
    exec .venv-ml/bin/python -u "$R/code/train.py" --arm "${arms[$I]}" "${args[@]}"
    ;;
  fixed) exec .venv-ml/bin/python -u "$R/code/evaluate.py" fixed --shard "$I" --nshards 16 ;;
  closed) exec .venv-ml/bin/python -u "$R/code/evaluate.py" closed --shard "$I" --nshards 32 ;;
  *) exit 2 ;;
esac
