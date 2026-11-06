#!/bin/bash
set -euo pipefail
cd /path/to/phyre-unc
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
/path/to/research-assets/miniconda3/envs/phyre/bin/python -u results/memory_deploy_pilot_20260929/code/collect_deploy_pilot.py --root results/memory_deploy_pilot_20260929 --scores /mnt/meg/jjenchen/phyre-unc/deploy_pilot_20260929/scores --workers 12
