#!/bin/bash
set -e
cd /path/to/phyre-unc
source /path/to/research-assets/miniconda3/etc/profile.d/conda.sh
conda activate phyre
python pipeline/make_rollouts.py --setup ball_within_template --fold 0 --split train --tag _clusters --clusters 4 --cluster-size 8 --radius 0.08 --verify 20 --workers 24
conda deactivate
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
M=/mnt/meg/jjenchen/phyre-unc; P=$M/latents/levjepa_overlap_g32_pca128/pca.pt
while pgrep -f "dev_clusters --shard" >/dev/null; do sleep 30; done
for i in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=$i .venv-ml/bin/python pipeline/encode_latents.py --encoder levjepa --manifests ball_within_template_f0_train_clusters --shard $i --nshards 5 --pca $P --out $M/latents > logs/enc_trainclusters_$i.log 2>&1 & done
wait
.venv-ml/bin/python pipeline/build_pack.py --latents $M/latents/levjepa_overlap_g32_pca128 --manifests ball_within_template_f0_dev_clusters --name within_f0_dev_clusters_pca128
.venv-ml/bin/python pipeline/build_pack.py --latents $M/latents/levjepa_overlap_g32_pca128 --manifests ball_within_template_f0_train_clusters --name within_f0_train_clusters_pca128
echo ALL_DONE
