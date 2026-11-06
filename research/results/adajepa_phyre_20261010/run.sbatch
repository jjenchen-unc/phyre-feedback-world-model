#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
cd /path/to/phyre-unc
exec .venv-ml/bin/python -u results/adajepa_phyre_20261010/code/tta.py "$@"
