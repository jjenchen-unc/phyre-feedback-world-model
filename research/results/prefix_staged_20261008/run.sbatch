#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
cd /path/to/phyre-unc
.venv-ml/bin/python -u results/prefix_staged_20261008/code/prefix_staged.py "$@" --root /path/to/phyre-unc/results/prefix_staged_20261008
