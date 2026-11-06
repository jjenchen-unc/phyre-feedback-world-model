#!/bin/bash
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=phyre_check
#SBATCH --output=/path/to/phyre-unc/results/deploy_pool_updater_20261016/single_%j.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$PWD/results/deploy_pool_updater_20261016/code:$PWD/pipeline
exec .venv-ml/bin/python -u results/deploy_pool_updater_20261016/code/deploy_pool_updater.py "$@"
