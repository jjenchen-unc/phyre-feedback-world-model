#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
cd /path/to/phyre-unc
.venv-ml/bin/python results/stable_scorer_cross_20260930/code/summarize_stable_closed_loop.py --root results/stable_scorer_cross_20260930 --project /path/to/phyre-unc
