#!/bin/bash
set -euo pipefail
cd /path/to/phyre-unc
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1
P=/path/to/phyre-unc
M=/mnt/meg/jjenchen/phyre-unc
D=$P/results/memory_deploy_pilot_20260929
W=$M/runs/memory_deploy_pilot_20260929
OLD=$P/results/memory_redesign_20260928/data/dataset.json
PY=$P/.venv-ml/bin/python
$PY -u "$D/code/summarize_deploy_pilot.py" --root "$D" --runs "$W"
