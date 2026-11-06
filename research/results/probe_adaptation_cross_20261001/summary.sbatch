#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/probe_adaptation_cross_20261001
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
"$PY" -u "$D/code/summarize_probe_adaptation.py" --root "$D"
