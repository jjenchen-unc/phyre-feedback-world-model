#!/bin/bash
set -euo pipefail
cd /path/to/phyre-unc
.venv-ml/bin/python -u results/memory_deploy_pilot_20260929/code/check_deploy_initial_routes.py
