#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
cd /path/to/phyre-unc
P=/path/to/phyre-unc
M=/mnt/meg/jjenchen/phyre-unc
D=$P/results/stable_scorer_cross_20260930
PY=$P/.venv-ml/bin/python
common=(--pack "$M/packs/cross_f1_dev_shortlist1024_pca128" --predictor "$M/runs/cross_f1_base_s16r16_pca128/predictor.pt" --probe "$P/results/probes_pca128_cross/ball_cross_template_f1_full.pt")
names=(base initial_R0 A_s0 B_s0 A_s1 B_s1)
name=${names[$((SLURM_ARRAY_TASK_ID / 2))]}
mode=legacy
if (( SLURM_ARRAY_TASK_ID % 2 )); then mode=stable_log; fi
mem=()
if [[ "$name" == initial_R0 ]]; then
  mem=(--memory "$M/runs/memory_redesign_20260928/R0/memory.pt" --feedback)
elif [[ "$name" != base ]]; then
  mem=(--memory "$M/runs/memory_deploy_pilot_20260929/$name/memory.pt" --feedback)
fi
"$PY" -u "$D/code/closed_loop_stable_scores.py" "${common[@]}" "${mem[@]}" --max-updates 10 --keep 4 --score-mode "$mode" --out "$D/${name}_${mode}.json"
