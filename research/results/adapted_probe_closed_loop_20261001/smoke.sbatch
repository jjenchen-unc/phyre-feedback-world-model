#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/adapted_probe_closed_loop_20261001
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
cd "$D/code"
sha256sum -c SHA256SUMS
"$PY" -u "$D/code/adapted_probe_closed_loop.py" run --root "$D" --index "$SLURM_ARRAY_TASK_ID" --smoke
