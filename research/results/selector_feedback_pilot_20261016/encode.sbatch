#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=phyre_fb_encode
#SBATCH --output=/path/to/phyre-unc/results/selector_feedback_pilot_20261016/encode_%A_%a.out
set -euo pipefail
cd /path/to/phyre-unc
export XDG_CACHE_HOME=/path/to/research-assets/xdg_dirs/cache
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH=$PWD/results/selector_feedback_pilot_20261016/code:$PWD/pipeline
record_failure() {
  failure_code=$?
  if [[ "$failure_code" != 0 ]]; then
    .venv-ml/bin/python - "$failure_code" <<'FAIL'
import json,os,sys,time
from pathlib import Path
p=Path('results/selector_feedback_pilot_20261016/FAILED.json')
p.write_text(json.dumps({'job':os.environ.get('SLURM_JOB_ID'),'array':os.environ.get('SLURM_ARRAY_TASK_ID'),'exit_code':int(sys.argv[1]),'time':time.time()}))
FAIL
  fi
}
trap record_failure EXIT
case "${CUDA_VISIBLE_DEVICES:-}" in
  0|1|2|3|4|5) ;;
  *) echo "Refusing allocation outside physical GPUs 0-5" >&2; exit 1 ;;
esac
if nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "Refusing occupied GPU" >&2; exit 1
fi
.venv-ml/bin/python -u results/selector_feedback_pilot_20261016/code/selector_feedback_data.py encode
.venv-ml/bin/python - <<'END'
import json,time
from pathlib import Path
p=Path('results/selector_feedback_pilot_20261016/ENCODING_COMPLETE.json');p.write_text(json.dumps({'complete':True,'time':time.time()}))
END
