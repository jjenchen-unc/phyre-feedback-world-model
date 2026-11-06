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
$PY -u "$D/code/build_pack.py" --latents "$M/deploy_pilot_20260929/latents/levjepa_overlap_g32_pca128" --manifests memory_deploy_pilot_20260929_rollouts --name memory_deploy_pilot_20260929
$PY -u "$D/code/audit_deploy_pilot.py" --root "$D" --old-data "$OLD"
