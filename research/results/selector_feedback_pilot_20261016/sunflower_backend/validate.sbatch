#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=phyre_rpc_validate
#SBATCH --output=/path/to/phyre-unc/results/selector_feedback_pilot_20261016/sunflower_backend/validate_%j.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH=$PWD/results/selector_feedback_pilot_20261016/code:$PWD/pipeline
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DISABLE_ADDMM_CUDA_LT
B=results/selector_feedback_pilot_20261016/sunflower_backend
.venv-ml/bin/python -u "$B/validate_rpc.py"
.venv-ml/bin/python -u "$B/runtime.py" hardware --panel cross
.venv-ml/bin/python -u "$B/runtime.py" hardware --panel within
