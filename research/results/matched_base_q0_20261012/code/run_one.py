"""One registered training/evaluation pair per Slurm job; four dependency lanes."""
import argparse
from pathlib import Path
import runpy
import torch

ap=argparse.ArgumentParser()
ap.add_argument('--panel',required=True,choices=['cross','within'])
ap.add_argument('--arch',required=True,choices=['base','q0'])
ap.add_argument('--weight',required=True,type=float)
ap.add_argument('--seed',required=True,type=int)
a=ap.parse_args()
torch.set_num_threads(4)
x=runpy.run_path(str(Path(__file__).with_name('matched_base_q0.py')))
plan=x['checked']()
assert a.weight in plan['weights'] and a.seed in plan['seeds_by_weight'][f'{a.weight:g}']
smoke=x['m'].read(x['ROOT']/f'SMOKE_{a.panel}.json')
assert smoke['passed'] and smoke['plan']==x['m'].sha(x['ROOT']/'PLAN.json')
x['train'](plan,a.panel,a.arch,a.weight,a.seed)
x['evaluate'](plan,a.panel,a.arch,a.weight,a.seed)
