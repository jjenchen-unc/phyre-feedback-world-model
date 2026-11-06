#!/bin/bash
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=phyre_pool
#SBATCH --output=/path/to/phyre-unc/results/deploy_pool_updater_20261016/data_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH=$PWD/results/deploy_pool_updater_20261016/code:$PWD/pipeline
if [[ "$1" == verify ]]; then
  panels=(cross within)
  exec .venv-ml/bin/python -u results/deploy_pool_updater_20261016/code/deploy_pool_data.py verify --panel "${panels[$SLURM_ARRAY_TASK_ID]}"
else
  exec .venv-ml/bin/python -u results/deploy_pool_updater_20261016/code/deploy_pool_data.py collect --shard "$SLURM_ARRAY_TASK_ID" --nshards 40
fi
