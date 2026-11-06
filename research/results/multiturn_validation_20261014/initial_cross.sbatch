#!/bin/bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/path/to/phyre-unc/pipeline
exec .venv-ml/bin/python -u results/multiturn_validation_20261014/code/benchmark_multiturn_initial.py --panel cross --output results/multiturn_validation_20261014/initial_cross.json
