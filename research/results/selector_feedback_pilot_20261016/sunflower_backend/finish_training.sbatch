#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=phyre_updater_finish
#SBATCH --output=/path/to/phyre-unc/results/selector_feedback_pilot_20261016/sunflower_backend/finish_%j.out
set -euo pipefail
cd /path/to/phyre-unc
exec .venv-ml/bin/python results/selector_feedback_pilot_20261016/sunflower_backend/finish_training.py
