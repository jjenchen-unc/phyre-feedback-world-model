#!/bin/bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/path/to/phyre-unc/pipeline
exec .venv-ml/bin/python -u results/multiturn_validation_20261014/eval_v2_code/validate_multiturn_runtime.py --mode eval --panel cross --output results/multiturn_validation_20261014/eval_v2_cross.json
