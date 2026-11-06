#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%16
#SBATCH --time=00:30:00
#SBATCH --job-name=phyre_plain_base
#SBATCH --output=/path/to/phyre-unc/results/plain_base_comparison_20261019/run_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
panels=(cross within)
panel=${panels[$((SLURM_ARRAY_TASK_ID % 2))]}
if [[ "${1:-}" == "smoke" ]]; then
 exec .venv-ml/bin/python -u results/plain_base_comparison_20261019/code/evaluate.py run --panel "$panel" --smoke
fi
exec .venv-ml/bin/python -u results/plain_base_comparison_20261019/code/evaluate.py run --panel "$panel" --shard "$((SLURM_ARRAY_TASK_ID / 2))" --nshards 8
