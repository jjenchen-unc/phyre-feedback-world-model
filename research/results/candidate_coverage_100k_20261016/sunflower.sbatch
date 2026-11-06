#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=phyre_cov100k
#SBATCH --output=/path/to/phyre-unc/results/candidate_coverage_100k_20261016/sunflower_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH=$PWD/results/candidate_coverage_100k_20261016/code:$PWD/pipeline
if [[ "$1" == smoke ]]; then
  panels=(cross within)
  exec .venv-ml/bin/python -u results/candidate_coverage_100k_20261016/code/candidate_coverage_100k.py smoke --method "$2" --panel "${panels[$SLURM_ARRAY_TASK_ID]}"
else
  exec .venv-ml/bin/python -u results/candidate_coverage_100k_20261016/code/candidate_coverage_100k.py run --method "$2" --shard "$SLURM_ARRAY_TASK_ID" --shards "$3"
fi
