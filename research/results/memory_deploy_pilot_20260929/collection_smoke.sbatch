#!/bin/bash
set -euo pipefail
cd /path/to/phyre-unc/results/memory_deploy_pilot_20260929/code
export PYTHONNOUSERSITE=1
/path/to/research-assets/miniconda3/envs/phyre/bin/python -u - <<'CODE'
import json,time
from collect_deploy_pilot import work
start=time.time()
r=work(('00001:000', '/path/to/phyre-unc/results/memory_deploy_pilot_20260929', '/mnt/meg/jjenchen/phyre-unc/deploy_pilot_20260929/scores', '/path/to/phyre-unc/data'))
print(json.dumps({'task':r['task'],'paired_episodes':len(r['episodes']),'rollouts':len(r['manifest']),'seconds':round(time.time()-start)}),flush=True)
CODE
