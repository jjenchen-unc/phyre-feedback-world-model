#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --job-name=phyre_milestone
#SBATCH --output=/path/to/phyre-unc/results/milestone_memory_20261015/master_%j.out
set -euo pipefail
cd /path/to/phyre-unc
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
exec .venv-ml/bin/python -u results/milestone_memory_20261015/code/launch_milestone_memory.py
