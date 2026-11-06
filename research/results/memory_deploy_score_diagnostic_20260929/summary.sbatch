#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=md_score_sum
#SBATCH --output=/path/to/phyre-unc/logs/md_score_sum_%j.out
set -euo pipefail
cd /path/to/phyre-unc
.venv-ml/bin/python results/memory_deploy_score_diagnostic_20260929/code/analyze_deploy_scores.py --root results/memory_deploy_score_diagnostic_20260929
