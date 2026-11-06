"""Filesystem-only compatibility fix; keep registered numerical code/weights immutable."""
import os
import sys
from pathlib import Path
ROOT=Path('/path/to/phyre-unc/results/query_token_staged_20261005')
sys.path.insert(0,str(ROOT/'code'))
import query_token_staged as s
m=s.checked(ROOT)
index=int(os.environ['SLURM_ARRAY_TASK_ID']);assert 0<=index<6
seed=index%3;within=index>=3
specs=s.f.read(ROOT/'within_inputs.json')['specs'] if within else [r for r in m['final_panels'] if r['panel']=='cross_natural']
assert len(specs)==100
for spec in specs:
    folder=ROOT/('within_evaluation' if within else 'evaluation')/f'q0_s{seed}'/'closed'/spec['task'].replace(':','_')
    folder.mkdir(parents=True,exist_ok=True)
print('STATIC_OUTPUT_DIRS_READY',seed,within,len(specs),flush=True)
