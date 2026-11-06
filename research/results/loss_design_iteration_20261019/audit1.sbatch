#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3
#SBATCH --time=00:10:00
#SBATCH --job-name=phyre_loss_audit1
#SBATCH --output=/path/to/phyre-unc/results/loss_design_iteration_20261019/audit1_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
R=/path/to/phyre-unc/results/loss_design_iteration_20261019
exec .venv-ml/bin/python -u "$R/code/dispatch.py" audit1 "$SLURM_ARRAY_TASK_ID"
