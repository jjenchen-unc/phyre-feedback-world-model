#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --array=0-4
#SBATCH --job-name=md_scores
#SBATCH --output=/path/to/phyre-unc/logs/md_scores_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
names=(initial_R0 A_s0 B_s0 A_s1 B_s1)
.venv-ml/bin/python -u results/memory_deploy_score_diagnostic_20260929/code/diagnose_deploy_scores.py --root results/memory_deploy_pilot_20260929 --out results/memory_deploy_score_diagnostic_20260929 --name "${names[$SLURM_ARRAY_TASK_ID]}"
