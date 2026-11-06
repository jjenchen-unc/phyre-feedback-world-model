#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/memory_redesign_20260928
W=/mnt/meg/jjenchen/phyre-unc/runs/memory_redesign_20260928
"${P}/.venv-ml/bin/python" -u "$D/code/summarize_memory_redesign.py" --out "$D" --runs "$W"
