"""Audit all saved original/Q0 static rankings and compute paired task outcomes."""
import json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
import prefix_staged as p
f=p.f

def main():
 m=p.checked(R);out={};budgets=[1,3,5,9,20];originals={}
 for split in ['cross','within']:
  out[split]={};originals[split]={}
  for seed in range(3):
   folder=R/'evaluation'/split/f'q0_s{seed}';complete=f.read(folder/'complete.json')
   assert complete['n']==100 and complete['provenance']['manifest']==f.digest(R/'manifest.json')
   tasks={};aucs={'original':[],'q0':[]}
   for spec in m['evaluation'][split+'_specs']:
    task=spec['task'];path=folder/task.replace(':','_');d=p.audit_case(path)
    with np.load(path/'scores.npz') as z:
     assert len(z['indices'])==1024 and np.isfinite(z['q0']).all() and np.isfinite(z['original']).all()
     np.testing.assert_array_equal(z['indices'],spec['indices'])
     if seed==0:originals[split][task]={k:z[k].copy() for k in ['indices','labels','original']}
     else:
      for k in ['indices','labels','original']:np.testing.assert_array_equal(z[k],originals[split][task][k])
     labels=z['labels'];n=int(labels.sum());ar={}
     for key in aucs:
      if 0<n<len(labels):
       pos=z[key][labels][:,None];neg=z[key][~labels][None,:]
       ar[key]=float(((pos>neg)+.5*(pos==neg)).mean());aucs[key].append(ar[key])
      else:ar[key]=None
     tasks[task]={'base_rank':d['original_first_success'],'q0_rank':d['first_success'],'positives':n,'auc':ar}
   def solved(rank,k):return rank is not None and rank<=k
   metrics={}
   for k in budgets:
    b={t:solved(v['base_rank'],k) for t,v in tasks.items()};q={t:solved(v['q0_rank'],k) for t,v in tasks.items()}
    metrics[f's{k}']={'base':sum(b.values()),'q0':sum(q.values()),'gained':sum(q[t] and not b[t] for t in tasks),
     'lost':sum(b[t] and not q[t] for t in tasks),'both':sum(b[t] and q[t] for t in tasks)}
    assert metrics[f's{k}']['q0']==complete['counts'][f's{k}']
   template={}
   for tp in sorted({t[:5] for t in tasks}):
    rr=[v for t,v in tasks.items() if t.startswith(tp)]
    template[tp]={'n':len(rr),'base_s9':sum(solved(v['base_rank'],9) for v in rr),'q0_s9':sum(solved(v['q0_rank'],9) for v in rr)}
   out[split][f's{seed}']={'metrics':metrics,'auc_defined_scenes':len(aucs['q0']),
    'auc_means':{k:float(np.mean(v)) for k,v in aucs.items()},'per_template':template,'per_task':tasks}
 for split in out:
  out[split]['aggregate']={k:{'base_percent':out[split]['s0']['metrics'][k]['base'],
   'q0_mean_percent':float(np.mean([out[split][f's{s}']['metrics'][k]['q0'] for s in range(3)])),
   'q0_per_seed':[out[split][f's{s}']['metrics'][k]['q0'] for s in range(3)]} for k in ['s1','s3','s5','s9','s20']}
  print(split,json.dumps(out[split]['aggregate']))
  print('auc',[out[split][f's{s}']['auc_means'] for s in range(3)],'n',out[split]['s0']['auc_defined_scenes'])
  print('paired_s9',[out[split][f's{s}']['metrics']['s9'] for s in range(3)])
 f.write_json(R/'stage1_comparison/RANKING.json',{'complete':True,'raw_score_audit_passed':True,
  'provenance':f.provenance(R),'source_script_sha256':f.digest(__file__),
  'scope':'Same full100 task IDs per split,1024 actions each,original versus finalQ0,same frozen predicted_mix_s0 scorer;3 seeds,no state updates',
  'splits':out})
if __name__=='__main__':main()
