#!/bin/bash
set -euo pipefail
cd /path/to/phyre-unc
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1
.venv-ml/bin/python -u results/memory_deploy_pilot_20260929/code/encode_latents.py --encoder levjepa --manifests memory_deploy_pilot_20260929_smoke --pca /mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32_pca128/pca.pt --out /mnt/meg/jjenchen/phyre-unc/deploy_pilot_20260929/latents
