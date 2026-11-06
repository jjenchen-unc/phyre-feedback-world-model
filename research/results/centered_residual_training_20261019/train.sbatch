#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=phyre_center_train
#SBATCH --output=/path/to/phyre-unc/results/centered_residual_training_20261019/train_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
R=/path/to/phyre-unc/results/centered_residual_training_20261019
panels=(cross within)
export PHYRE_ENCODER_ENDPOINT="$R/encoder${SLURM_ARRAY_TASK_ID}/rpc/ENDPOINT.json"
exec .venv-ml/bin/python -u "$R/code/train.py" train --panel "${panels[$SLURM_ARRAY_TASK_ID]}" --kind centered
