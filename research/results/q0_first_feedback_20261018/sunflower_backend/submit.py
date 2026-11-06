"""Validation -> four fresh trainings -> automatic matched evaluation -> audit."""
import json
import os
from pathlib import Path
import subprocess
import time

B = Path(__file__).resolve().parent
R = B.parent
P = B.parents[2]
os.chdir(P)
assert not (B / 'JOBS.json').exists(), 'Already submitted; inspect jobs before retrying'
assert json.loads((R / 'DATA_AUDIT.json').read_text())['passed']
assert (B / 'rpc/ENDPOINT.json').exists() and not (B / 'rpc/STOP').exists()
jobs = {'time': time.time(), 'scope': '4 training arms, then 800 method-task evaluations'}


def submit(name, script, dependency=None, array=None):
    cmd = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes']
    if dependency:
        cmd.append('--dependency=' + dependency)
    if array:
        cmd.append('--array=' + array)
    job = subprocess.check_output(cmd + [str(B / script)], text=True).strip()
    jobs[name] = job
    (B / 'JOBS.json').write_text(json.dumps(jobs, indent=2) + '\n')
    print('SUBMITTED', name, job, flush=True)
    return job


validate = submit('validation', 'validate.sbatch')
preflight = submit('preflight', 'preflight.sbatch', 'afterok:' + validate, '0-3')
train = submit('training', 'train.sbatch', 'afterok:' + preflight, '0-3')
finish_train = submit('training_audit', 'finish_training.sbatch', 'afterok:' + train)
evaluation = submit('evaluation', 'evaluate.sbatch', 'afterok:' + finish_train, '0-31%16')
submit('evaluation_audit', 'finish_evaluation.sbatch', 'afterany:' + evaluation)
(R / 'STATUS.json').write_text(json.dumps({'stage': 'validation_queued', 'jobs': jobs,
    'training_complete': False, 'evaluation_complete': False}, indent=2) + '\n')
