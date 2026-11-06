#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/support_utility_20261004
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
cd "$D/code"
sha256sum -c SHA256SUMS
"$PY" -m unittest discover -s . -p test_support_utility.py -v
for arm in uniform near mixed; do
  "$PY" -u support_utility.py train --root "$D" --arm "$arm" --seed 0 --smoke
  "$PY" -u support_utility.py eval --root "$D" --arm "$arm" --seed 0 --smoke
done
"$PY" -u support_utility.py summary --root "$D" --smoke
