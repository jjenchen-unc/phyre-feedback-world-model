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
$PY -u "$D/code/encode_latents.py" --encoder levjepa --manifests memory_deploy_pilot_20260929_rollouts --pca "$M/latents/levjepa_overlap_g32_pca128/pca.pt" --out "$M/deploy_pilot_20260929/latents" --shard $SLURM_ARRAY_TASK_ID --nshards 4
