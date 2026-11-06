#!/bin/bash
set -euo pipefail
cd /path/to/phyre-unc
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1
.venv-ml/bin/python -u results/memory_deploy_pilot_20260929/code/score_deploy_pilot.py --candidates /mnt/meg/jjenchen/phyre-unc/candidates/memory_deploy_pilot_20260929 --out /mnt/meg/jjenchen/phyre-unc/deploy_pilot_20260929/scores --shard $SLURM_ARRAY_TASK_ID --nshards 4
