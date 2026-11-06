#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/support_utility_20261004
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
cd "$D/code"
sha256sum -c SHA256SUMS
"$PY" -u support_utility.py summary --root "$D"
