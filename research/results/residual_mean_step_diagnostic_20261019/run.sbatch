#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%10
#SBATCH --time=02:00:00
#SBATCH --job-name=phyre_mean_step
#SBATCH --output=/path/to/phyre-unc/results/residual_mean_step_diagnostic_20261019/log_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
panels=(cross cross cross cross cross within within within within within)
exec .venv-ml/bin/python -u results/residual_mean_step_diagnostic_20261019/code/diagnose.py run --panel "${panels[$SLURM_ARRAY_TASK_ID]}" --shard "$((SLURM_ARRAY_TASK_ID % 5))" --nshards 5
