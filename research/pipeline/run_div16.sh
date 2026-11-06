#!/bin/bash
set -e
source /path/to/research-assets/miniconda3/etc/profile.d/conda.sh
conda activate phyre
cd /path/to/phyre-unc
for setup in ball_within_template ball_cross_template; do
  python pipeline/make_rollouts.py --setup $setup --fold 0 --split train --tasks-from ${setup}_f0_train_sub400 --tag _sub400_div16 --n-success 8 --n-fail 8 --diverse --verify 10 --workers 24
done
echo ALL_DONE
