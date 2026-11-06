#!/bin/bash
set -e
cd /path/to/phyre-unc
source /path/to/research-assets/miniconda3/etc/profile.d/conda.sh
conda activate phyre
python pipeline/render_scenes.py --setup ball_cross_template --fold 1
python pipeline/render_candidates.py --manifest ball_cross_template_f1_dev_n100
python pipeline/truth_rule_candidates.py --candidates /mnt/meg/jjenchen/phyre-unc/candidates/ball_cross_template_f1_dev_n100 --rule soft 4 2 --out /mnt/meg/jjenchen/phyre-unc/cl_truth_rule_cross_f1_dev > logs/truth_cross_f1.log 2>&1 &
conda deactivate
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
CUDA_VISIBLE_DEVICES=5 .venv-ml/bin/python pipeline/scene_classifier.py --setup ball_cross_template_f1 --candidates /mnt/meg/jjenchen/phyre-unc/candidates/ball_cross_template_f1_dev_n100 --data-mode all --updates 100000 --save-scores /mnt/meg/jjenchen/phyre-unc/cl_scores_sceneclf_cross_f1 --out results/scene_clf_cross_f1.json
wait
echo ALL_DONE
