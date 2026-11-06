#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/support_within_20261004
PH=/path/to/research-assets/miniconda3/envs/phyre/bin/python
export OMP_NUM_THREADS=1
cd "$D/code"
sha256sum -c SHA256SUMS
"$PH" -u within_support_data.py candidates --root "$D" --workers 12
