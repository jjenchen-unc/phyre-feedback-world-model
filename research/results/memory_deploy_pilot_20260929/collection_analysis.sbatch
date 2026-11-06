#!/bin/bash
set -euo pipefail
cd /path/to/phyre-unc
.venv-ml/bin/python -u results/memory_deploy_pilot_20260929/code/analyze_deploy_collection.py --root results/memory_deploy_pilot_20260929 --old-data results/memory_redesign_20260928/data/dataset.json
