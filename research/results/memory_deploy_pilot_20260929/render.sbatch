#!/bin/bash
set -euo pipefail
cd /path/to/phyre-unc
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
/path/to/research-assets/miniconda3/envs/phyre/bin/python -u results/memory_deploy_pilot_20260929/code/render_candidates.py --manifest memory_deploy_pilot_20260929 --pool 5000 --workers 12
