#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
cd /path/to/phyre-unc
if [[ "$1" == "labels" ]]; then
 exec /path/to/research-assets/miniconda3/envs/phyre/bin/python -u results/small_official_20261010/recovery_20261010/data_fixed.py labels
fi
exec .venv-ml/bin/python -u results/small_official_20261010/recovery_20261010/recovery_runner.py "$@"
