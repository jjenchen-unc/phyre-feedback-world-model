#!/bin/bash
set -e
source /path/to/research-assets/miniconda3/etc/profile.d/conda.sh
conda activate phyre
cd /path/to/phyre-unc
for setup in ball_within_template ball_cross_template; do
  python pipeline/make_rollouts.py --setup $setup --split dev --max-tasks 100 --verify 20 --workers 16
  python pipeline/make_rollouts.py --setup $setup --split train --verify 20 --workers 16
done
echo ALL_DONE
