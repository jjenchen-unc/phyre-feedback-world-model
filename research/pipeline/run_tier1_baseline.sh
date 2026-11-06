#!/bin/bash
set -e
cd /path/to/phyre-unc
M=/mnt/meg/jjenchen/phyre-unc
until [ $(ls $M/latents/levjepa_overlap_g32_pca128/meta_shard*of4.json 2>/dev/null | xargs -r grep -l dev_clusters | wc -l) -ge 4 ]; do sleep 30; done
.venv-ml/bin/python pipeline/build_pack.py --latents $M/latents/levjepa_overlap_g32_pca128 --manifests ball_within_template_f0_dev_clusters --name within_f0_dev_clusters_pca128
cd pipeline && CUDA_VISIBLE_DEVICES=4 ../.venv-ml/bin/python eval_tier1.py --predictor $M/runs/base_s16r16_pca128/predictor.pt --out ../results/tier1_baseline_within_f0.json
echo ALL_DONE
