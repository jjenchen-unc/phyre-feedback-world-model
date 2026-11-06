#!/bin/bash
set -e
cd /path/to/phyre-unc/pipeline
R=/mnt/meg/jjenchen/phyre-unc/runs
until [ -f $R/mem_v1_full/memory.pt ]; do sleep 60; done
B=$R/base_s16r16_pca128/predictor.pt
CUDA_VISIBLE_DEVICES=5 ../.venv-ml/bin/python eval_tier1.py --predictor $B --memory $R/mem_v1_full/memory.pt --out ../results/tier1_mem_v1_full.json > ../logs/eval_mem_v1_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 ../.venv-ml/bin/python eval_tier1.py --predictor $B --memory $R/mem_v1_full/memory.pt --empty-memory --out ../results/tier1_mem_v1_full_empty.json > ../logs/eval_mem_v1_full_empty.log 2>&1 &
wait
echo ALL_DONE
