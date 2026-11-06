#!/bin/bash
set -e
cd /path/to/phyre-unc/pipeline
NAME=$1; TRAIN=$2; shift 2
R=/mnt/meg/jjenchen/phyre-unc/runs; B=$R/base_s16r16_pca128/predictor.pt
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES start $(date)"
cat /mnt/meg/jjenchen/phyre-unc/packs/${TRAIN}_latents.npy > /dev/null
../.venv-ml/bin/python train_memory.py --mode residual --steps 20000 --train $TRAIN --out $R/$NAME "$@" > ../logs/$NAME.log 2>&1
../.venv-ml/bin/python eval_tier1.py --predictor $B --memory $R/$NAME/memory.pt --out ../results/tier1_$NAME.json > ../logs/eval_$NAME.log 2>&1
../.venv-ml/bin/python eval_tier1.py --predictor $B --memory $R/$NAME/memory.pt --empty-memory --out ../results/tier1_${NAME}_empty.json > ../logs/eval_${NAME}_empty.log 2>&1
echo "ALL_DONE $(date)"
