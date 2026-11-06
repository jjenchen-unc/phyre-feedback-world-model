#!/bin/bash
set -e
cd /path/to/phyre-unc/pipeline
R=/mnt/meg/jjenchen/phyre-unc/runs; B=$R/base_s16r16_pca128/predictor.pt
CUDA_VISIBLE_DEVICES=5 ../.venv-ml/bin/python train_memory.py --mode residual --steps 20000 --out $R/mem_v1_residual > ../logs/mem_v1_residual.log 2>&1
CUDA_VISIBLE_DEVICES=5 ../.venv-ml/bin/python eval_tier1.py --predictor $B --memory $R/mem_v1_residual/memory.pt --out ../results/tier1_mem_v1_residual.json > ../logs/eval_mem_v1_residual.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 ../.venv-ml/bin/python eval_tier1.py --predictor $B --memory $R/mem_v1_residual/memory.pt --empty-memory --out ../results/tier1_mem_v1_residual_empty.json > ../logs/eval_mem_v1_residual_empty.log 2>&1 &
wait
echo ALL_DONE
