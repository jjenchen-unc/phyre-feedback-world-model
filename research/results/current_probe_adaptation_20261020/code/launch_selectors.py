import json
import os
from pathlib import Path
import subprocess
import time
P = Path('/path/to/phyre-unc')
R = P/'results/current_probe_adaptation_20261020'
for shard, gpu in enumerate([4, 5]):
    path = R/f'selector_launch_{shard}.json'
    assert not path.exists()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), XDG_CACHE_HOME='/path/to/research-assets/xdg_dirs/cache',
               HF_HUB_OFFLINE='1', PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    with (R/f'selector_{shard}.out').open('w') as f:
        proc = subprocess.Popen(['timeout', '--kill-after=30', '7200', str(P/'.venv-ml/bin/python'),
                                 '-u', str(R/'code/score_scenes.py'), '--shard', str(shard)],
                                env=env, cwd=P, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    path.write_text(json.dumps(dict(pid=proc.pid, gpu=gpu, started=time.time(), timeout_s=7200))+'\n')
    print('SELECTOR_STARTED', shard, proc.pid, flush=True)
