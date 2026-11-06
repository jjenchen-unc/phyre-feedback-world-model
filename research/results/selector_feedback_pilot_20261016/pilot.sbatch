#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --gres=gpu:6
#SBATCH --time=48:00:00
#SBATCH --job-name=phyre_fb_pilot
#SBATCH --output=/path/to/phyre-unc/results/selector_feedback_pilot_20261016/pilot_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH=$PWD/results/selector_feedback_pilot_20261016/code:$PWD/pipeline
exec .venv-ml/bin/python -u results/selector_feedback_pilot_20261016/code/selector_feedback_launch.py pilot
