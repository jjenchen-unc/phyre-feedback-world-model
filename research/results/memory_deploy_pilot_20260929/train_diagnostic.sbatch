#!/bin/bash
set -euo pipefail
cd /path/to/phyre-unc
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1
V=R${SLURM_ARRAY_TASK_ID}
.venv-ml/bin/python -u results/memory_deploy_pilot_20260929/code/eval_memory_redesign.py --data results/memory_deploy_pilot_20260929/train_panel.json --memory /mnt/meg/jjenchen/phyre-unc/runs/memory_redesign_20260928/$V/memory.pt --out results/memory_deploy_pilot_20260929/${V}_train_fixed.json
