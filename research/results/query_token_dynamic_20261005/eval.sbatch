#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4
cd /path/to/phyre-unc/results/query_token_dynamic_20261005/code
sha256sum -c SHA256SUMS
/path/to/phyre-unc/.venv-ml/bin/python -u query_token_dynamic.py eval --root /path/to/phyre-unc/results/query_token_dynamic_20261005 --index "$SLURM_ARRAY_TASK_ID"
