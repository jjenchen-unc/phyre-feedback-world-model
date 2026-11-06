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
for V in A B; do
$PY -u "$D/code/train_deploy_pilot.py" --root "$D" --old-data "$OLD" --out "$W/smoke_$V" --variant "$V" --seed 0 --steps 12
done
$PY -u "$D/code/eval_deploy_pilot.py" --root "$D" --memory "$M/runs/memory_redesign_20260928/R0/memory.pt" --out "$D/initial_R0_fixed.json"
