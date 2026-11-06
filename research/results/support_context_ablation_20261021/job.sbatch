#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
set -euo pipefail
cd /path/to/phyre-unc
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
R=results/support_context_ablation_20261021
if [ "$1" = prepare ]; then
 exec .venv-ml/bin/python -u "$R/study.py" prepare
fi
exec .venv-ml/bin/python -u "$R/study.py" run --index "$SLURM_ARRAY_TASK_ID"
