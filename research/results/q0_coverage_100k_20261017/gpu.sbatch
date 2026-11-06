#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --job-name=phyre_q0_100k
#SBATCH --output=/path/to/phyre-unc/results/q0_coverage_100k_20261017/job_%A_%a.out
#SBATCH --gres=gpu:1
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH=$PWD/results/q0_coverage_100k_20261017/code:$PWD/pipeline
unset CUBLAS_WORKSPACE_CONFIG DISABLE_ADDMM_CUDA_LT
if [[ "$1" == smoke ]]; then
  panels=(cross within)
  exec .venv-ml/bin/python -u results/q0_coverage_100k_20261017/code/q0_coverage_100k.py smoke --panel "${panels[$SLURM_ARRAY_TASK_ID]}"
else
  exec .venv-ml/bin/python -u results/q0_coverage_100k_20261017/code/q0_coverage_100k.py run --shard "$SLURM_ARRAY_TASK_ID" --shards 40
fi
