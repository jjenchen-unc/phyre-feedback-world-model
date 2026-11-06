#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=phyre_upd_preflight
#SBATCH --output=/path/to/phyre-unc/results/selector_feedback_pilot_20261016/sunflower_backend/preflight_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH=$PWD/results/selector_feedback_pilot_20261016/code:$PWD/pipeline
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
panels=(cross cross within within)
kinds=(residual zero residual zero)
exec .venv-ml/bin/python -u results/selector_feedback_pilot_20261016/sunflower_backend/runtime.py preflight --panel "${panels[$SLURM_ARRAY_TASK_ID]}" --kind "${kinds[$SLURM_ARRAY_TASK_ID]}"
