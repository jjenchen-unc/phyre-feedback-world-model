#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=phyre_eval_summary
#SBATCH --output=/path/to/phyre-unc/results/q0_first_feedback_20261018/sunflower_backend/eval_finish_%j.out
set -euo pipefail
cd /path/to/phyre-unc
export PYTHONPATH=$PWD/results/q0_first_feedback_20261018/code:$PWD/pipeline
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
unset DISABLE_ADDMM_CUDA_LT
exec .venv-ml/bin/python -u results/q0_first_feedback_20261018/sunflower_backend/finish_evaluation.py
