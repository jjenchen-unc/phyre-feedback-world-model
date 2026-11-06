#!/bin/bash
set -euo pipefail
P=/path/to/phyre-unc
D=$P/results/memory_redesign_20260928
W=/mnt/meg/jjenchen/phyre-unc/runs/memory_redesign_20260928
V=R${SLURM_ARRAY_TASK_ID}
"$P/.venv-ml/bin/python" -u "$D/code/train_memory_redesign.py" --data "$D/data/dataset.json" --out "$W/$V" --variant "$V" --steps 20000
"$P/.venv-ml/bin/python" -u "$D/code/eval_memory_redesign.py" --data "$D/data/dataset.json" --memory "$W/$V/memory.pt" --out "$D/${V}_fixed.json"
"$P/.venv-ml/bin/python" -u "$D/code/closed_loop_memory.py" --pack /mnt/meg/jjenchen/phyre-unc/packs/cross_f1_dev_shortlist1024_pca128 --predictor /mnt/meg/jjenchen/phyre-unc/runs/cross_f1_base_s16r16_pca128/predictor.pt --probe "$P/results/probes_pca128_cross/ball_cross_template_f1_full.pt" --memory "$W/$V/memory.pt" --feedback --max-updates 10 --keep 4 --out "$D/${V}_closed_loop.json"
