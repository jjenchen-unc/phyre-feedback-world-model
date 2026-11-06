#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --job-name=phyre_transfer_smoke
#SBATCH --output=/path/to/phyre-unc/results/support_transfer_iteration_20261019/smoke_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
R=/path/to/phyre-unc/results/support_transfer_iteration_20261019
L=/path/to/phyre-unc/results/loss_design_iteration_20261019
export PHYRE_ENCODER_ENDPOINT="$L/encoder$((SLURM_ARRAY_TASK_ID))/rpc/ENDPOINT.json"
exec .venv-ml/bin/python -u "$R/code/transfer_dispatch.py" smoke "$SLURM_ARRAY_TASK_ID"
