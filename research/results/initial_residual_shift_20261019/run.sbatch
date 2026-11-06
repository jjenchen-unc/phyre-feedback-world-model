#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
#SBATCH --time=00:30:00
#SBATCH --job-name=phyre_res_shift
#SBATCH --output=/path/to/phyre-unc/results/initial_residual_shift_20261019/run_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
exec .venv-ml/bin/python -u results/initial_residual_shift_20261019/code/diagnose.py run --shard "$SLURM_ARRAY_TASK_ID" --nshards 4
