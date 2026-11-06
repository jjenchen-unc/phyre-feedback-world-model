#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=phyre_encoder
#SBATCH --output=/path/to/phyre-unc/results/deploy_pool_updater_20261016/encoder_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=$PWD/results/deploy_pool_updater_20261016/code:$PWD/pipeline
.venv-ml/bin/python - <<'PY'
import os, subprocess, torch
assert torch.cuda.device_count()==1
gpu=str(torch.cuda.get_device_properties(0).uuid)
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
others=[line for line in busy.splitlines() if line.split(',')[0].strip()==gpu and int(line.split(',')[1])!=os.getpid()]
assert not others, ('Allocated GPU already occupied; refusing to share with another job',others)
PY
exec .venv-ml/bin/python -u results/deploy_pool_updater_20261016/code/deploy_encoder_service.py serve
