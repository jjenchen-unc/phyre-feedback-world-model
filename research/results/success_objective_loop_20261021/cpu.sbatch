#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
set -euo pipefail
cd /path/to/phyre-unc
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec .venv-ml/bin/python -u "$@"
