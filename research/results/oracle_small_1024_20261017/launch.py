"""Bounded Oprime physical GPU4/5 diagnostic; never touches other users' jobs."""
from pathlib import Path
import json
import os
import subprocess
import time

P = Path('/path/to/phyre-unc')
R = P / 'results/oracle_small_1024_20261017'
os.chdir(P)


def write(name, value):
    tmp = R / (name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(R / name)


def available(gpus):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    used = {int(a): int(b) for a, b in (r.split(',') for r in rows.splitlines())}
    assert all(used[g] < 100 for g in gpus), used


def start(mode, gpu, seconds, name):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), XDG_CACHE_HOME='/path/to/research-assets/xdg_dirs/cache',
               HF_HUB_OFFLINE='1', PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               PYTHONPATH=str(P / 'pipeline'))
    env.pop('CUBLAS_WORKSPACE_CONFIG', None); env.pop('DISABLE_ADDMM_CUDA_LT', None)
    with (R / name).open('a') as log:
        return subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=60', str(seconds),
                                 str(P / '.venv-ml/bin/python'), '-u', str(R / 'code/oracle_small_1024.py'), mode],
                                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)


try:
    available([4, 5])
    smoke = start('smoke', 4, 1800, 'smoke.out')
    write('STATUS.json', {'stage': 'smoke', 'pid': smoke.pid, 'controller': os.getpid(), 'timestamp': time.time()})
    code = smoke.wait()
    if code:
        raise RuntimeError(f'Smoke failed, exit {code}; formal workers not launched')
    available([4, 5])
    workers = [start('worker', g, 16 * 3600, f'worker_{g}.out') for g in [4, 5]]
    write('STATUS.json', {'stage': 'running', 'pids': [w.pid for w in workers], 'physical_gpus': [4, 5],
                          'controller': os.getpid(), 'timeout_hours': 16, 'timestamp': time.time()})
    codes = [w.wait() for w in workers]
    subprocess.run([str(P / '.venv-ml/bin/python'), str(R / 'code/oracle_small_1024.py'), 'summary'], check=True)
    complete = json.loads((R / 'SUMMARY.json').read_text())['complete']
    write('STATUS.json', {'stage': 'complete' if complete and codes == [0, 0] else 'incomplete',
                          'exit_codes': codes, 'complete': complete, 'timestamp': time.time()})
except Exception as exc:
    write('STATUS.json', {'stage': 'failed', 'error': str(exc), 'timestamp': time.time()})
    raise
