"""Exact static-memory policy from complete saved first-update score vectors."""
from transfer_core import *
import numpy as np

def run():
 plan=read(S/'PLAN.json');out={'complete':True,'scope':'Exact static-policy replay, frozen after first failed action update. No future observations used for ranking; outcomes only stop the simulated attempt sequence. GPU replay validation still required.','panels':{}}
 for panel,spec in plan['panels'].items():
  methods={}
  for mode in ['raw','centered']:
   base=B/'evaluation_128'/panel/'uniform' if mode=='raw' else C/'evaluation'/panel/'uniform';first={};full={}
   for task in spec['tasks']:
    source=base/task.replace(':','_');row=read(source/'result.json')
    with np.load(source/'scores.npz') as z,np.load(S/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as q:
     scores=z['scores'];labels=z['labels'];indices=z['indices'];np.testing.assert_array_equal(scores[0],q['scores'][0]);assert int(scores[0].argmax())==int(z['selected'][0]);j=int(z['selected'][0]);selected=[j];success=1 if labels[j] else None
     if success is None:
      assert len(scores)>1 and np.isfinite(scores[1]).all();v=scores[1].copy();v[j]=-np.inf
      for k in range(8):
       j=int(v.argmax());v[j]=-np.inf;selected.append(j)
       if labels[j]:success=k+2;break
     assert len(selected)==len(set(selected));first[task]=success;full[task]=row['first_success']
     write(R/'freeze'/panel/mode/task.replace(':','_')/'result.json',{'complete':True,'task':task,'first_success':success,'selected':selected,'action_idx':indices[selected].tolist(),'source_scores_sha256':sha(source/'scores.npz'),'source_result_sha256':sha(source/'result.json')})
   methods[mode+'_first_only']=metrics(first);methods[mode+'_full']=metrics(full)
  out['panels'][panel]={'methods':methods,'paired':{mode+'_full_vs_first_only':paired(methods[mode+'_full'],methods[mode+'_first_only']) for mode in ['raw','centered']}}
 write(R/'ROUND0_FREEZE.json',out);print(json.dumps({p:{k:v['counts'] for k,v in d['methods'].items()} for p,d in out['panels'].items()},indent=2))
if __name__=='__main__':run()
