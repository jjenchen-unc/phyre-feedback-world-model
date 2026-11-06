#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
R=/path/to/phyre-unc/results/success_objective_loop_20261021
I=${SLURM_ARRAY_TASK_ID:-0}
case "$1" in
 pack) exec .venv-ml/bin/python -u "$R/code/study.py" pack --shard "$I" --nshards 8 ;;
 smoke) exec .venv-ml/bin/python -u "$R/code/train.py" smoke ;;
 train) exec .venv-ml/bin/python -u "$R/code/train.py" train --index "$I" ;;
 discovery|confirmation) exec .venv-ml/bin/python -u "$R/code/evaluate.py" "$1" --shard "$I" --nshards 8 ;;
 *) exit 2 ;;
esac
