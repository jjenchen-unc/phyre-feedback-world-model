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
mkdir -p "$D/smoke"
for name in base initial_R0; do
  mem=()
  if [[ "$name" == initial_R0 ]]; then mem=(--memory "$M/runs/memory_redesign_20260928/R0/memory.pt" --feedback); fi
  for mode in legacy stable_log; do
    "$PY" -u "$D/code/closed_loop_stable_scores.py" "${common[@]}" "${mem[@]}" --max-updates 1 --keep 4 --tasks 00014:273 --score-mode "$mode" --out "$D/smoke/${name}_${mode}.json"
  done
done
"$PY" "$D/code/check_smoke.py" --root "$D" --project "$P"
