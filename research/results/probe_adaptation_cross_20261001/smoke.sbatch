#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/probe_adaptation_cross_20261001
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
"$PY" -u "$D/code/probe_adaptation.py" cache --root "$D" --smoke
for arm in true_control predicted_mix initial_only; do
 "$PY" -u "$D/code/probe_adaptation.py" train --root "$D" --arm "$arm" --seed 0 --smoke
done
"$PY" -u "$D/code/probe_adaptation.py" eval --root "$D" --smoke
"$PY" -u "$D/code/summarize_probe_adaptation.py" --root "$D" --smoke
