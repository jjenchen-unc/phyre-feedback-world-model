import json
from pathlib import Path
from collections import Counter
import numpy as np
R=Path('/path/to/phyre-unc/results/milestone_memory_20261015')
s=json.loads((R/'SUMMARY.json').read_text());output={'method':'Post-hoc replay of stored full scores; no new model training or GPU inference','panels':{}}
def first(scores,labels,mask):
 ids=np.flatnonzero(mask);order=ids[np.argsort(-scores[ids],kind='stable')];hit=np.flatnonzero(labels[order]);return int(hit[0]+1) if len(hit) else None
def solved(x,b):return x is not None and x<=b
for panel,arms in s['panels'].items():
 out=output['panels'][panel]={};training={}
 for kind in ['residual','zero']:
  rows=[json.loads(l) for l in (R/'models'/panel/kind/'episodes.jsonl').read_text().splitlines()]
  ev=[e for row in rows for e in row['events']];counts=Counter(e['used_updates'] for e in ev)
  assert all(e['solved'][0] for e in ev if e['used_updates']==0)
  training[kind]={'terminal_depth_counts':dict(sorted(counts.items())),'episodes_reaching_update':{k:sum(e['used_updates']>=k for e in ev) for k in [1,2,3,4,5,6,7,8]},'slots':len(rows),'optimizer_steps':sum(r['optimizer_step'] for r in rows),'caps':dict(Counter(e['cap'] for e in ev)),'last_1000':{'depths':dict(Counter(e['used_updates'] for r in rows[-1000:] for e in r['events']))}}
 out['training']=training
 for kind in ['residual','zero','tta']:
  diagnostics=arms[kind]['first_update'];defined=[x for x in diagnostics if x['best_success_rank_before'] is not None];auc=[x['auc_after']-x['auc_before'] for x in diagnostics if x['auc_before'] is not None]
  d=out[kind]={'first_update':{'n_updated':len(diagnostics),'n_with_success_candidate':len(defined),'rank_better':sum(x['best_success_rank_after']<x['best_success_rank_before'] for x in defined),'rank_equal':sum(x['best_success_rank_after']==x['best_success_rank_before'] for x in defined),'rank_worse':sum(x['best_success_rank_after']>x['best_success_rank_before'] for x in defined),'mean_auc_delta':float(np.mean(auc)),'median_auc_delta':float(np.median(auc)),'auc_improved':sum(x>0 for x in auc),'auc_worsened':sum(x<0 for x in auc)} }
  cap_results={k:{} for k in range(9)};turns={};divergence=[];n_success_candidates=[]
  for task in arms[kind]['first_success']:
   f=R/'evaluation'/panel/kind/task.replace(':','_');rec=json.loads((f/'result.json').read_text())
   with np.load(f/'scores.npz') as z:scores=z['scores'];labels=z['labels'];selected=z['selected'];indices=z['indices']
   n_success_candidates.append(int(labels.sum()))
   for k in range(9):
    if rec['first_success'] is not None and rec['first_success']<=k:fs=rec['first_success']
    else:
     assert k<len(scores)
     mask=np.ones(len(labels),bool);mask[selected[:k]]=False
     rank=first(scores[k],labels,mask);fs=None if rank is None else k+rank
    cap_results[k][task]=fs
   assert cap_results[0][task]==arms['q0']['first_success'][task]
   assert solved(cap_results[8][task],9)==solved(rec['first_success'],9)
   for k in range(1,len(scores)):
    mask=np.ones(len(labels),bool);mask[selected[:k]]=False
    before=first(scores[k-1],labels,mask);after=first(scores[k],labels,mask)
    if before is None:continue
    turns.setdefault(k,[]).append({'task':task,'before':before,'after':after,'remaining':9-k})
  d['update_cap_replay']={k:{'counts':{b:sum(solved(x,b) for x in vals.values()) for b in [1,3,5,9]},'paired_full9':{'freeze_only':[t for t in vals if solved(vals[t],9) and not solved(cap_results[8][t],9)],'full_only':[t for t in vals if not solved(vals[t],9) and solved(cap_results[8][t],9)]}} for k,vals in cap_results.items()}
  d['eval_episodes_reaching_update']={k:sum(len(json.loads(p.read_text())['events'])>k for p in (R/'evaluation'/panel/kind).glob('*/result.json')) for k in range(1,9)}
  d['per_update_rank']={k:{'n':len(v),'better':sum(x['after']<x['before'] for x in v),'worse':sum(x['after']>x['before'] for x in v),'equal':sum(x['after']==x['before'] for x in v),'into_remaining_budget':sum(x['after']<=x['remaining']<x['before'] for x in v),'out_of_remaining_budget':sum(x['before']<=x['remaining']<x['after'] for x in v)} for k,v in turns.items()}
  d['pool']={'with_success':sum(x>0 for x in n_success_candidates),'median_success_count':float(np.median(n_success_candidates))}
(R/'DIAGNOSTICS.json').write_text(json.dumps(output,indent=2)+'\n')
for panel,d in output['panels'].items():
 print('\nPANEL',panel,'TRAIN',json.dumps(d['training']))
 for kind in ['residual','zero','tta']:
  a=d[kind];print(kind,'FIRST',a['first_update'],'DEPTH',a['eval_episodes_reaching_update'],'POOL',a['pool'])
  print('CAP REPLAY',[(k,c['counts']) for k,c in a['update_cap_replay'].items()]);print('PER_UPDATE',a['per_update_rank'])
