#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=scene_baseline
#SBATCH --output=/path/to/phyre-unc/results/milestone_scene_baseline_20261015/eval_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
scene_gpu_id=${SLURM_JOB_GPUS%%,*}
case "$scene_gpu_id" in [0-5]) ;; *) echo "Unexpected GPU allocation: $scene_gpu_id"; exit 1 ;; esac
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$scene_gpu_id" --query-gpu=uuid --format=csv,noheader)
scene_panel=cross
if [ "$SLURM_ARRAY_TASK_ID" = 1 ]; then scene_panel=within; fi
exec .venv-ml/bin/python -u results/milestone_scene_baseline_20261015/code/milestone_scene_baseline.py eval --panel "$scene_panel"
