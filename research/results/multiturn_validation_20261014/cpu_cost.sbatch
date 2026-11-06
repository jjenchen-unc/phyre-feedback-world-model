#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /path/to/research-assets/miniconda3/envs/phyre/bin/python -u results/multiturn_validation_20261014/code/benchmark_multiturn_cpu.py
