#!/bin/bash
set -e
cd /path/to/phyre-unc/pipeline
R=/mnt/meg/jjenchen/phyre-unc/runs; B=$R/base_s16r16_pca128/predictor.pt
cat /mnt/meg/jjenchen/phyre-unc/packs/within_f0_train_clusters_pca128_latents.npy > /dev/null
CUDA_VISIBLE_DEVICES=0 ../.venv-ml/bin/python train_memory.py --train within_f0_train_clusters_pca128 --mode residual --steps 20000 --out $R/mem_v1_residual_clusters > ../logs/mem_v1_residual_clusters.log 2>&1
CUDA_VISIBLE_DEVICES=0 ../.venv-ml/bin/python eval_tier1.py --predictor $B --memory $R/mem_v1_residual_clusters/memory.pt --out ../results/tier1_mem_v1_residual_clusters.json > ../logs/eval_mem_v1_residual_clusters.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 ../.venv-ml/bin/python eval_tier1.py --predictor $B --memory $R/mem_v1_residual_clusters/memory.pt --empty-memory --out ../results/tier1_mem_v1_residual_clusters_empty.json > ../logs/eval_mem_v1_residual_clusters_empty.log 2>&1 &
wait
echo ALL_DONE
