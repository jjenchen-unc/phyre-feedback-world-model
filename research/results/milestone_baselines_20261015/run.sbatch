#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=30
#SBATCH --mem=160G
#SBATCH --time=36:00:00
#SBATCH --job-name=phyre_oracles
#SBATCH --output=/path/to/phyre-unc/results/milestone_baselines_20261015/master_%j.out
set -euo pipefail
cd /path/to/phyre-unc
export OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
exec .venv-ml/bin/python -u results/milestone_baselines_20261015/launch_milestone_baselines.py
