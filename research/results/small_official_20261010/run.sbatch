#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
cd /path/to/phyre-unc
if [[ "$1" == "cpu_labels" ]]; then
 exec /path/to/research-assets/miniconda3/envs/phyre/bin/python -u results/small_official_20261010/code/data_cpu.py labels
fi
if [[ "$1" == "smoke_train" ]]; then
 .venv-ml/bin/python results/small_official_20261010/code/test_prefix_policy_predictor.py
 .venv-ml/bin/python results/small_official_20261010/code/test_prefix_staged.py
fi
exec .venv-ml/bin/python -u results/small_official_20261010/code/run_small.py "$@"
