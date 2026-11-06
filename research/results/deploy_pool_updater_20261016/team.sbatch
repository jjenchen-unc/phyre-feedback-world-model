#!/bin/bash
#SBATCH --partition=defq
#SBATCH --ntasks=10
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=3
#SBATCH --time=7-00:00:00
#SBATCH --job-name=phyre_update
#SBATCH --output=/path/to/phyre-unc/results/deploy_pool_updater_20261016/team_%j.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$PWD/results/deploy_pool_updater_20261016/code:$PWD/pipeline
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))
srun --kill-on-bad-exit=1 .venv-ml/bin/python -u results/deploy_pool_updater_20261016/code/deploy_pool_updater.py "$@"
