"""Submit paired 15-model training/evaluation only after all GPU smoke checks pass."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/path/to/phyre-unc/results/query_token_dynamic_20261005')
sys.path.insert(0,str(ROOT/'code'))
import query_token_dynamic as q

m=q.checked(ROOT)
smoke=q.f.read(ROOT/'SMOKE.json')
assert smoke['passed'] and smoke['provenance']==q.f.provenance(ROOT)
jobs=q.f.read(ROOT/'jobs.json')
assert 'train' not in jobs, 'Do not duplicate a submitted experiment'


def submit(stage,array=None,dependencies=None,gpu=True,hours=12):
    cmd=['sbatch','--parsable','--partition=a6000','--nodelist=oprime.ib',
         '--cpus-per-task=4','--mem=32G',f'--time={hours:02d}:00:00',
         '--job-name=qtd_'+stage,'--output=/path/to/phyre-unc/logs/qtd_'+stage+'_%A_%a.out']
    if gpu:cmd.append('--gres=gpu:1')
    if array:cmd.append('--array='+array)
    if dependencies:cmd.append('--dependency=afterok:'+':'.join(str(i) for i in dependencies))
    cmd.append(str(ROOT/(stage+'.sbatch')))
    job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0])
    jobs[stage]=job
    q.f.write_json(ROOT/'jobs.json',jobs)
    print(stage,job,flush=True)
    return job


train=submit('train','0-14%5')
evaluation=submit('eval','0-14%5',[train],hours=24)
submit('summary',dependencies=[evaluation],gpu=False,hours=4)
within=submit('prepare_within',dependencies=[34413],gpu=False,hours=4)
within_eval=submit('within_eval','0-14%5',[train,evaluation,within],hours=24)
submit('within_summary',dependencies=[within_eval],gpu=False,hours=4)
