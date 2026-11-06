#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
cd /path/to/phyre-unc
RUN_ROOT=/path/to/phyre-unc/results/query_token_dynamic_full_eval_20261007
if [ "$1" = "eval" ]; then
  .venv-ml/bin/python -u "$RUN_ROOT/evaluate_full.py" eval --index "$SLURM_ARRAY_TASK_ID"
else
  .venv-ml/bin/python -u "$RUN_ROOT/evaluate_full.py" "$1"
fi
