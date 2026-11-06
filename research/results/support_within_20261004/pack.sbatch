#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/support_within_20261004
A=/mnt/meg/jjenchen/phyre-unc/support_within_20261004
PY=$P/.venv-ml/bin/python
export OMP_NUM_THREADS=4
cd "$D/code"
sha256sum -c SHA256SUMS
"$PY" -u build_pack.py --latents "$A/latents/levjepa_overlap_g32_pca128" --data "$A/data" --manifests within_support --name within_support_pca128 --out "$A/packs"
