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
I=$SLURM_ARRAY_TASK_ID
if (( I % 2 == 0 )); then V=A; else V=B; fi
S=$(( I / 2 ))
NAME=${V}_s${S}
$PY -u "$D/code/train_deploy_pilot.py" --root "$D" --old-data "$OLD" --out "$W/$NAME" --variant "$V" --seed "$S" --steps 2000
$PY -u "$D/code/eval_deploy_pilot.py" --root "$D" --memory "$W/$NAME/memory.pt" --out "$D/${NAME}_fixed.json"
