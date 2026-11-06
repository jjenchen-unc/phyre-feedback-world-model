#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/support_within_20261004
A=/mnt/meg/jjenchen/phyre-unc/support_within_20261004
PY=$P/.venv-ml/bin/python
PH=/path/to/research-assets/miniconda3/envs/phyre/bin/python
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
cd "$D/code"
sha256sum -c SHA256SUMS
"$PH" -u within_support_data.py candidates --root "$D" --workers 2 --smoke
"$PY" -u within_support.py prefilter --root "$D" --smoke
"$PH" -u within_support_data.py rollouts --root "$D" --workers 2 --smoke
"$PY" -u encode_latents.py --encoder levjepa --manifests within_support --data "$A/smoke_data" --out "$A/smoke_latents" --pca /mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32_pca128/pca.pt
"$PY" -u build_pack.py --latents "$A/smoke_latents/levjepa_overlap_g32_pca128" --data "$A/smoke_data" --manifests within_support --name within_support_pca128 --out "$A/smoke_packs"
"$PY" -u within_support.py base --root "$D" --smoke
"$PY" -u within_support.py prepare --root "$D" --smoke
"$PY" -u within_support.py eval --root "$D" --arm uniform --seed 0 --smoke
"$PY" -u within_support.py summary --root "$D" --smoke
