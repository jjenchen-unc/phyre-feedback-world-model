#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=phyre_feedback_eval
#SBATCH --output=/path/to/phyre-unc/results/selector_feedback_pilot_20261016/sunflower_backend/eval_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH=$PWD/results/selector_feedback_pilot_20261016/code:$PWD/pipeline
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
panels=(cross cross cross cross within within within within)
kinds=(q0 residual zero tta q0 residual zero tta)
arm=$((SLURM_ARRAY_TASK_ID % 8))
shard=$((SLURM_ARRAY_TASK_ID / 8))
exec .venv-ml/bin/python -u results/selector_feedback_pilot_20261016/sunflower_backend/runtime.py eval --panel "${panels[$arm]}" --kind "${kinds[$arm]}" --shard "$shard" --nshards 4
