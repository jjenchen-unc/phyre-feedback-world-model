#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/support_within_20261004
A=/mnt/meg/jjenchen/phyre-unc/support_within_20261004
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
cd "$D/code"
sha256sum -c SHA256SUMS
"$PY" -u encode_latents.py --encoder levjepa --manifests within_support --data "$A/data" --out "$A/latents" --pca /mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32_pca128/pca.pt --shard "$SLURM_ARRAY_TASK_ID" --nshards 2
