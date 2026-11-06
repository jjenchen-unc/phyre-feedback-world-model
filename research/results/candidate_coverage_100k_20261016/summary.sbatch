#!/bin/bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=phyre_covsum
#SBATCH --output=/path/to/phyre-unc/results/candidate_coverage_100k_20261016/summary_%j.out
set -euo pipefail
cd /path/to/phyre-unc
exec .venv-ml/bin/python -u results/candidate_coverage_100k_20261016/code/candidate_coverage_100k.py summary --wait
