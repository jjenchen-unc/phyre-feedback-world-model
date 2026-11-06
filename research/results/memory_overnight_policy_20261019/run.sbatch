#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=phyre_policy
#SBATCH --output=/path/to/phyre-unc/results/memory_overnight_policy_20261019/log_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
export PHYRE_ENCODER_ENDPOINT=/path/to/phyre-unc/results/memory_overnight_policy_20261019/encoder$((SLURM_ARRAY_TASK_ID % 2))/rpc/ENDPOINT.json
if [[ "${1:-}" == smoke ]]; then
  panels=(cross within)
  exec .venv-ml/bin/python -u results/memory_overnight_policy_20261019/code/policy.py run --panel "${panels[$SLURM_ARRAY_TASK_ID]}" --variant real
fi
panels=(cross cross cross cross within within within within)
variants=(step_0.25 step_0.5 centered feedback_only step_0.25 step_0.5 centered feedback_only)
arm=$((SLURM_ARRAY_TASK_ID % 8))
shard=$((SLURM_ARRAY_TASK_ID / 8))
exec .venv-ml/bin/python -u results/memory_overnight_policy_20261019/code/policy.py run --panel "${panels[$arm]}" --variant "${variants[$arm]}" --shard "$shard" --nshards 4
