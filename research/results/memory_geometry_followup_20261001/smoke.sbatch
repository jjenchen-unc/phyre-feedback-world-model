#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/memory_geometry_followup_20261001
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
cd "$D/code"
sha256sum -c SHA256SUMS
for arm in latent_only latent_geometry; do
  "$PY" -u "$D/code/memory_geometry_followup.py" train --root "$D" --arm "$arm" --seed 0 --smoke
  "$PY" -u "$D/code/memory_geometry_followup.py" eval --root "$D" --arm "$arm" --seed 0 --smoke
done
"$PY" -u "$D/code/memory_geometry_followup.py" summary --root "$D" --smoke
