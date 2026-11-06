import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--project',type=Path);a=p.parse_args()
ref=json.loads((a.project/'results/cx_clm_base.json').read_text());expected=next(r['first_success'] for r in ref['per_task'] if r['task']=='00014:273')
outputs={};scores={}
for name in ['base','initial_R0']:
 for mode in ['legacy','stable_log']:
  f=a.root/'smoke'/f'{name}_{mode}.json';v=json.loads(f.read_text());outputs[name,mode]=v
  assert v['aggregate']['n_tasks']==1
  with np.load(str(f)+'.tasks/00014_273_scores.npz') as z: scores[name,mode]=z['scores'].copy()
  assert np.isfinite(scores[name,mode]).all()
  s=scores[name,mode][:,1];assert np.array_equal(np.argsort(s,axis=1,kind='stable'),np.argsort(np.exp(s),axis=1,kind='stable'))
assert outputs['base','legacy']['per_task'][0]['first_success']==expected
for mode in ['legacy','stable_log']:
 assert np.array_equal(scores['base',mode][0],scores['initial_R0',mode][0])
for name in ['base','initial_R0']:
 assert np.array_equal(scores[name,'legacy'],scores[name,'stable_log'])
result={'passed':True,'task':'00014:273','historical_base_first_success':expected,'empty_identity':True,'stable_log_direct_order':True,'paired_score_channels_identical_for_one_update':True}
(a.root/'smoke_check.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
