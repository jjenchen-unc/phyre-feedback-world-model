#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4
cd /path/to/phyre-unc/results/query_token_staged_20261005/code
sha256sum -c SHA256SUMS
/path/to/phyre-unc/.venv-ml/bin/python -u /path/to/phyre-unc/results/query_token_staged_20261005/dispatch.py q0_eval
