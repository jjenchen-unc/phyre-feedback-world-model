#!/bin/bash
set -e
cd /path/to/phyre-unc
M=/mnt/meg/jjenchen/phyre-unc; L=$M/latents/levjepa_overlap_g32_pca128; P=$L/pca.pt
source /path/to/research-assets/miniconda3/etc/profile.d/conda.sh
conda activate phyre
python pipeline/make_rollouts.py --setup ball_cross_template --fold 1 --split train --tag _s16r16 --n-success 16 --n-fail 0 --n-random 16 --diverse --verify 20 --workers 24
conda deactivate
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
while pgrep -f "train_clusters --shard" >/dev/null; do sleep 60; done
for i in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=$i .venv-ml/bin/python pipeline/encode_latents.py --encoder levjepa --manifests ball_cross_template_f1_train_s16r16 ball_cross_template_f1_dev_n100 --shard $i --nshards 6 --pca $P --out $M/latents > logs/enc_cross_f1_$i.log 2>&1 & done
wait
.venv-ml/bin/python pipeline/build_pack.py --latents $L --manifests ball_cross_template_f1_train_s16r16 --name cross_f1_train_s16r16_pca128
.venv-ml/bin/python pipeline/build_pack.py --latents $L --manifests ball_cross_template_f1_dev_n100 --name cross_f1_dev_pca128
.venv-ml/bin/python pipeline/make_subset.py --manifest ball_cross_template_f1_train_s16r16 --tasks 400
CUDA_VISIBLE_DEVICES=0 .venv-ml/bin/python pipeline/train_predictor.py --train cross_f1_train_s16r16_pca128 --dev cross_f1_dev_pca128 --epochs 8 --out $M/runs/cross_f1_base_s16r16_pca128 > logs/pred_cross_f1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python pipeline/probe_rule.py --latents $L --setups ball_cross_template --fold 1 --train-tags _s16r16_sub400 --grids full --calib template_cv --save-probe results/probes_pca128_cross --out results/probe_cross_f1_pca128.json > logs/probe_cross_f1_pca128.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 .venv-ml/bin/python pipeline/closed_loop_eval.py --candidates $M/candidates/ball_cross_template_f1_dev_n100 --predictor $M/runs/cross_f1_base_s16r16_pca128/predictor.pt --probe results/probes_pca128_cross/ball_cross_template_f1_full.pt --pca $P --save-scores $M/cl_scores_cross_f1_base --out results/cl_cross_f1_base.json
.venv-ml/bin/python pipeline/cl_combine.py --clf $M/cl_scores_sceneclf_cross_f1 --wm $M/cl_scores_cross_f1_base --out results/combine_cross_f1.json
echo ALL_DONE
