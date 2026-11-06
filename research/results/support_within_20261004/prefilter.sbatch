#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/support_within_20261004
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
cd "$D/code"
sha256sum -c SHA256SUMS
"$PY" -u within_support.py prefilter --root "$D" --shard "$SLURM_ARRAY_TASK_ID" --nshards 2
