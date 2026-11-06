#!/bin/bash
set -e
source /path/to/research-assets/miniconda3/etc/profile.d/conda.sh
conda activate phyre
cd /path/to/phyre-unc
for fold in 1 2; do
  for setup in ball_within_template ball_cross_template; do
    python pipeline/make_rollouts.py --setup $setup --fold $fold --split dev --max-tasks 100 --verify 20 --workers 24
    python pipeline/make_rollouts.py --setup $setup --fold $fold --split train --verify 20 --workers 24
  done
done
echo ALL_DONE
