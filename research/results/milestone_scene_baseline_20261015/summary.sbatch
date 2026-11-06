#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=scene_summary
#SBATCH --output=/path/to/phyre-unc/results/milestone_scene_baseline_20261015/summary_%j.out
set -euo pipefail
cd /path/to/phyre-unc
export OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
exec .venv-ml/bin/python -u results/milestone_scene_baseline_20261015/code/milestone_scene_baseline.py summary
