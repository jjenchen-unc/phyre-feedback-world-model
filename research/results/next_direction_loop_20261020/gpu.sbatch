#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
R=/path/to/phyre-unc/results/next_direction_loop_20261020
I=${SLURM_ARRAY_TASK_ID:-0}
case "$1" in
  smoke_train) exec .venv-ml/bin/python -u "$R/code/train_head.py" --index "$I" --smoke ;;
  smoke_eval) exec .venv-ml/bin/python -u "$R/code/evaluate_heads.py" smoke ;;
  train) exec .venv-ml/bin/python -u "$R/code/train_head.py" --index "$I" ;;
  discovery|confirmation) exec .venv-ml/bin/python -u "$R/code/evaluate_heads.py" "$1" --shard "$I" --nshards 8 ;;
  *) exit 2 ;;
esac
