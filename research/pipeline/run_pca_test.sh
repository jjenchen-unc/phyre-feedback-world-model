#!/bin/bash
set -e
cd /path/to/phyre-unc
CUDA_VISIBLE_DEVICES=2 .venv-ml/bin/python pipeline/pca_project.py
for k in 256 128; do
  CUDA_VISIBLE_DEVICES=2 .venv-ml/bin/python pipeline/probe_rule.py --latents /mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32_pca$k --setups ball_within_template --fold 0 --train-tags _sub400_div16 --grids full --calib task --out results/pca_probe_within_k$k.json &
done
wait
echo ALL_DONE
