#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/probe_adaptation_cross_20261001
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
arms=(true_control predicted_mix initial_only)
arm=${arms[$((SLURM_ARRAY_TASK_ID % 3))]}
seed=$((SLURM_ARRAY_TASK_ID / 3))
"$PY" -u "$D/code/probe_adaptation.py" train --root "$D" --arm "$arm" --seed "$seed"
