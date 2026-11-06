#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-31%32
#SBATCH --time=02:00:00
#SBATCH --job-name=phyre_center_eval
#SBATCH --output=/path/to/phyre-unc/results/query_balance_centered_eval_20261019/eval_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
export PHYRE_ENCODER_ENDPOINT=/path/to/phyre-unc/results/memory_overnight_policy_20261019/encoder$((SLURM_ARRAY_TASK_ID % 2))/rpc/ENDPOINT.json
if [[ "${1:-}" == "smoke" ]]; then
    panels=(cross within)
    exec .venv-ml/bin/python -u results/query_balance_centered_eval_20261019/code/evaluate.py run --panel "${panels[$SLURM_ARRAY_TASK_ID]}" --kind uniform --smoke
fi
panels=(cross cross within within)
kinds=(uniform balanced uniform balanced)
arm=$((SLURM_ARRAY_TASK_ID % 4))
shard=$((SLURM_ARRAY_TASK_ID / 4))
exec .venv-ml/bin/python -u results/query_balance_centered_eval_20261019/code/evaluate.py run --panel "${panels[$arm]}" --kind "${kinds[$arm]}" --shard "$shard" --nshards 8
