"""Independent NumPy reconstruction of final static/dynamic policies."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def solved(rank,k):return rank is not None and rank<=k
m=read(R/'manifest.json');summary=read(R/'SUMMARY.json')
assert summary['complete'] and summary['raw_audit_passed'] and summary['provenance']['manifest']==sha(R/'manifest.json')
result={};cases=0
for split,ss in summary['splits'].items():
 ranks={};reference={}
 for seed in range(3):
  ranks[f'original_s{seed}']={}
  for stage in ['q0','residual','zero']:
   name=f'{stage}_s{seed}';ranks[name]={};folder=R/'evaluation'/split/name
   done=read(folder/'complete.json');train=read(R/'models'/name/'complete.json')
   assert done['provenance']['checkpoint']==train['checkpoint_sha256']
   assert done['provenance']['manifest']==sha(R/'manifest.json')
   for spec in m['evaluation'][split+'_specs']:
    task=spec['task'];path=folder/task.replace(':','_');d=read(path/'result.json')
    assert d['provenance']==done['provenance'] and d['task']==task
    with np.load(path/'scores.npz') as z:
     np.testing.assert_array_equal(z['indices'],spec['indices']);labels=z['labels'];assert len(labels)==1024
     if stage=='q0':
      assert np.isfinite(z['q0']).all() and np.isfinite(z['original']).all()
      for k in ['q0','original']:
       hits=np.flatnonzero(labels[np.argsort(-z[k],kind='stable')]);rank=int(hits[0])+1 if len(hits) else None
       expected=d['first_success'] if k=='q0' else d['original_first_success'];assert rank==expected
       ranks[f'{k}_s{seed}'][task]=rank
      if seed==0:reference[task]={k:z[k].copy() for k in ['indices','labels','original']}
      else:
       for k in reference[task]:np.testing.assert_array_equal(z[k],reference[task][k])
     else:
      baseline=R/'evaluation'/split/f'q0_s{seed}'/task.replace(':','_')/'scores.npz'
      with np.load(baseline) as q0:
       np.testing.assert_array_equal(z['indices'],q0['indices']);np.testing.assert_array_equal(labels,q0['labels'])
       np.testing.assert_array_equal(z['scores'][0],q0['q0'])
      assert z['scores'].shape==(d['attempts'],1024) and len(z['selected'])==d['attempts']
      seen=set();rank=None
      for step,(scores,chosen) in enumerate(zip(z['scores'],z['selected']),1):
       assert np.isfinite(scores).all()
       order=np.argsort(-scores,kind='stable');expected=next(int(i) for i in order if int(i) not in seen)
       assert expected==int(chosen);seen.add(expected)
       if labels[expected]:assert step==d['attempts'];rank=step
      assert rank==d['first_success'] and d['updates']==d['attempts']-1
      assert rank is not None or d['attempts']==d['budget']==20
      ranks[name][task]=rank
    cases+=1
   counts={f's{k}':sum(solved(v,k) for v in ranks[name].values()) for k in [1,3,5,9,20]}
   assert len(ranks[name])==100 and counts==done['counts']==ss[name]['counts']
   assert ranks[name]==ss[name]['first_success']
  assert ranks[f'original_s{seed}']==ranks['original_s0']
  for stage in ['residual','zero']:
   assert {t for t,v in ranks[f'{stage}_s{seed}'].items() if v==1}=={t for t,v in ranks[f'q0_s{seed}'].items() if v==1}
 averages={}
 for stage in ['original','q0','residual','zero']:
  averages[stage]={f's{k}':{'mean_percent':float(np.mean([sum(solved(v,k) for v in ranks[f'{stage}_s{s}'].values()) for s in range(3)])),
    'per_seed':[sum(solved(v,k) for v in ranks[f'{stage}_s{s}'].values()) for s in range(3)]} for k in [1,3,5,9,20]}
 paired={}
 for ref in ['original','q0','zero']:
  paired['residual_minus_'+ref]={}
  for k in [9,20]:
   vv=[]
   for seed in range(3):
    a=ranks[f'residual_s{seed}'];b=ranks[f'{ref}_s{seed}']
    win=sum(solved(a[t],k) and not solved(b[t],k) for t in a);loss=sum(solved(b[t],k) and not solved(a[t],k) for t in a)
    vv.append({'seed':seed,'gained':win,'lost':loss,'net_pp':win-loss})
   paired['residual_minus_'+ref][f's{k}']=vv
 result[split]={'averages':averages,'paired':paired}
 print(split,json.dumps(averages),json.dumps(paired),flush=True)
assert cases==1800
out={'passed':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'raw_cases':cases,'summary_sha256':sha(R/'SUMMARY.json'),'script_sha256':sha(Path(__file__)),
 'checks':['raw static ranking','original scores identical across seeds','dynamic stable argmax over untried actions',
 'labels and success stopping','each updater first scores equal same-seed Q0','all saved first-success and aggregate counts'],
 'splits':result,'limits':'Descriptive results on reused dev panels; no independent test, significance or pure-architecture causal claim'}
(R/'final_analysis/AUDIT.json').write_text(json.dumps(out,indent=2))
