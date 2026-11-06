#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=phyre_fb_enc_check
#SBATCH --output=/path/to/phyre-unc/results/selector_feedback_pilot_20261016/sunflower_backend/encoder_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH=$PWD/results/selector_feedback_pilot_20261016/code:$PWD/pipeline
exec .venv-ml/bin/python -u results/selector_feedback_pilot_20261016/sunflower_backend/check_encoder.py "$SLURM_ARRAY_TASK_ID"
