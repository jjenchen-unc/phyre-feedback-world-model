#!/bin/bash
#SBATCH --job-name=mr_eval_smoke
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/phyre-unc/logs/mr_eval_smoke_%j.out
set -euo pipefail
export OMP_NUM_THREADS=4
P=/path/to/phyre-unc/.venv-ml/bin/python
D=/path/to/phyre-unc/results/memory_redesign_20260928
W=/mnt/meg/jjenchen/phyre-unc/runs/memory_redesign_20260928
$P "$D/code/eval_memory_redesign.py" --data "$D/data/eval_smoke_dataset.json" --memory "$W/smoke_R4/memory.pt" --out "$D/smoke_fixed.json"
$P "$D/code/closed_loop_memory.py" --pack /mnt/meg/jjenchen/phyre-unc/packs/cross_f1_dev_shortlist1024_pca128 --predictor /mnt/meg/jjenchen/phyre-unc/runs/cross_f1_base_s16r16_pca128/predictor.pt --probe /path/to/phyre-unc/results/probes_pca128_cross/ball_cross_template_f1_full.pt --memory "$W/smoke_R4/memory.pt" --feedback --max-updates 10 --keep 4 --max-tasks 2 --out "$D/smoke_closed_loop.json"
