#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec .venv-ml/bin/python results/multiturn_validation_20261014/code/benchmark_repeated_simulation.py
