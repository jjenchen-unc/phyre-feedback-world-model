#!/bin/bash
set -e
cd /path/to/phyre-unc
source /path/to/research-assets/miniconda3/etc/profile.d/conda.sh
conda activate phyre
python pipeline/make_subset.py --manifest ball_cross_template_f1_train --tasks 400
python pipeline/make_rollouts.py --setup ball_cross_template --fold 1 --split train --tasks-from ball_cross_template_f1_train_sub400 --tag _sub400_div16 --n-success 8 --n-fail 8 --diverse --verify 10 --workers 24
conda deactivate
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
M="ball_cross_template_f1_train_sub400_div16 ball_cross_template_f1_dev_n100"
CUDA_VISIBLE_DEVICES=2 .venv-ml/bin/python pipeline/encode_latents.py --encoder levjepa --manifests $M --shard 0 --nshards 2 --out /mnt/meg/jjenchen/phyre-unc/latents > logs/enc_f1_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 .venv-ml/bin/python pipeline/encode_latents.py --encoder levjepa --manifests $M --shard 1 --nshards 2 --out /mnt/meg/jjenchen/phyre-unc/latents > logs/enc_f1_1.log 2>&1
wait
CUDA_VISIBLE_DEVICES=2 .venv-ml/bin/python pipeline/probe_rule.py --latents /mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32 --setups ball_cross_template --fold 1 --train-tags _sub400_div16 --grids full --calib template_cv --save-probe results/probes --save-preds results/preds_f1_cross_tcv --out results/tcv_f1_cross.json
echo ALL_DONE
