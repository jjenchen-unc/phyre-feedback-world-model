#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4
cd /path/to/phyre-unc/results/memory_fresh_geometry_20261005/code
sha256sum -c SHA256SUMS
/path/to/phyre-unc/.venv-ml/bin/python -u fresh_memory_geometry.py train --root /path/to/phyre-unc/results/memory_fresh_geometry_20261005 --seed "$SLURM_ARRAY_TASK_ID"
