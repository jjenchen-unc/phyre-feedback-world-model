#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=phyre_cov100k
#SBATCH --output=/path/to/phyre-unc/results/candidate_coverage_100k_20261016/oprime_%A_%a.out
set -euo pipefail
case "${CUDA_VISIBLE_DEVICES:-}" in
  0|1|2|3|4|5) ;;
  *) echo "Refusing allocation outside own available physical GPUs 0-5" >&2; exit 1 ;;
esac
if nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "Refusing GPU with an existing compute process" >&2
  exit 1
fi
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
