#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=36
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=phyre_continue
#SBATCH --output=/path/to/phyre-unc/results/deploy_pool_updater_20261016/continue_%j.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=$PWD/results/deploy_pool_updater_20261016/code:$PWD/pipeline
exec .venv-ml/bin/python -u results/deploy_pool_updater_20261016/continue_oprime.py
