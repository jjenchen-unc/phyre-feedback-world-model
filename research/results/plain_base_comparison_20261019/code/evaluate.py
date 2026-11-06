"""Plain base on the current classifier top1024 pools, no feedback or encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
P=Path(__file__).resolve().parents[3]
R=P/'results/plain_base_comparison_20261019'
S=P/'results/q0_first_feedback_20261018'
C=P/'results/query_balance_centered_eval_20261019'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def register():
 source=read(S/'PLAN.json')
 d={'code_sha256':sha(__file__),'source_plan_sha256':sha(S/'PLAN.json'),'tasks':{p:v['tasks'] for p,v in source['panels'].items()},'checkpoints':{p:v['initial']['base'] for p,v in source['panels'].items()},'budget':9,'protocol':'Same100scenes/protocol and classifier-selected1024pool. Plain base uses own first action and static ranking thereafter. Additional CPU control forces Q0 first action, then follows base ranking excluding it. No feedback, simulations, encoding or training.','smoke':'Bothpanels firstlistedtask: fullQ0scores exact vs source; plainbase fastscores exact vs legacy scoring; no memorytokens; fixedweights/head.','limits':'Deployment task/budget matched, not equal totaltrainingcompute. Base-vs-Q0 primary allows differentfirstaction; Q0-firstbase control separatelylabelled.'}
 if (R/'PLAN.json').exists():assert read(R/'PLAN.json')==d
 else:write(R/'PLAN.json',d)
 print('REGISTERED')
def orderings(scores,qfirst):
 order=np.argsort(-scores,kind='stable')
 return {'base':order,'base_q0_first':np.r_[qfirst,order[order!=qfirst]]}
def first_success(order,labels,budget=9):
 hits=np.flatnonzero(labels[order[:budget]])
 return int(hits[0]+1) if len(hits) else None
def run(panel,shard,nshards,smoke):
 import torch
 sys.path[:0]=[str(S/'code'),str(S/'sunflower_backend'),str(P/'pipeline')]
 import selector_feedback_run as src
 from fast_inference import CandidateRanker
 torch.set_num_threads(2)
 plan=read(R/'PLAN.json');assert sha(__file__)==plan['code_sha256'] and sha(S/'PLAN.json')==plan['source_plan_sha256']
 if not smoke:assert read(R/'SMOKE.json')['passed']
 source=src.data.checked();model,probe,head,rule=src.old.load_model(panel,'base',source)
 assert not hasattr(model,'initial_tokens') and not hasattr(model,'initial_state')
 fixed=src.old.ident(model);headhash=src.old.ident(head)
 tasks=plan['tasks'][panel][:1] if smoke else plan['tasks'][panel][shard::nshards]
 for task in tasks:
  folder=R/('smoke' if smoke else 'evaluation')/panel/task.replace(':','_')
  pool=src.data.load_pool(panel,task)
  with np.load(S/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as z:
   np.testing.assert_array_equal(pool['action_idx'],z['indices']);np.testing.assert_array_equal(pool['status']>0,z['labels']);qref=z['scores'][0].copy()
  z0=torch.from_numpy(pool['z0'].copy()).cuda().float();actions=torch.from_numpy(pool['action']).cuda();features=[f[None] for f in pool['features']]
  start=time.time();ranker=CandidateRanker(model,head,rule,z0,actions,features);scores=ranker.rank().scores;torch.cuda.synchronize();elapsed=time.time()-start
  if smoke:
   with torch.no_grad():reference=src.old.p.static_scores(model,z0,actions,features,src.old.p.score_function(head,rule))
   np.testing.assert_array_equal(scores,reference)
   qmodel,qprobe,qhead,qrule=src.old.load_model(panel,'q0',source)
   np.testing.assert_array_equal(CandidateRanker(qmodel,qhead,qrule,z0,actions,features).rank().scores,qref)
   assert src.old.ident(qhead)==headhash and qrule==rule
   del qmodel,qprobe,qhead
  assert src.old.ident(model)==fixed and src.old.ident(head)==headhash
  labels=pool['status']>0;qfirst=int(qref.argmax());orders=orderings(scores,qfirst)
  src.data.save_npz(folder/'scores.npz',scores=scores,indices=pool['action_idx'],labels=labels)
  write(folder/'result.json',{'complete':True,'panel':panel,'task':task,'plan_sha256':sha(R/'PLAN.json'),'checkpoint_sha256':plan['checkpoints'][panel]['sha256'],'model_unchanged':True,'head_unchanged':True,'no_memory_tokens':True,'q0_first_local':qfirst,'rank_seconds':elapsed,'smoke':smoke,'replay_exact':smoke,'first_success':{k:first_success(v,labels) for k,v in orders.items()}})
  print(panel,task,{k:first_success(v,labels) for k,v in orders.items()},round(elapsed,2),flush=True)
def summary(smoke):
 plan=read(R/'PLAN.json');assert sha(__file__)==plan['code_sha256']
 out={'complete':True,'plan_sha256':sha(R/'PLAN.json'),'verified_cases':0,'panels':{},'scope':plan['protocol'],'limits':plan['limits']}
 original=read(S/'SUMMARY.json');centered=read(C/'SUMMARY.json')
 for panel,alltasks in plan['tasks'].items():
  tasks=alltasks[:1] if smoke else alltasks
  methods={m:{'n':len(tasks),'first_success':{}} for m in ['base','base_q0_first']}
  for task in tasks:
   folder=R/('smoke' if smoke else 'evaluation')/panel/task.replace(':','_');row=read(folder/'result.json')
   assert row['complete'] and row['model_unchanged'] and row['head_unchanged'] and row['no_memory_tokens']
   assert row['plan_sha256']==sha(R/'PLAN.json') and row['checkpoint_sha256']==plan['checkpoints'][panel]['sha256'] and row['smoke']==smoke
   if smoke:assert row['replay_exact']
   with np.load(folder/'scores.npz') as z,np.load(S/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as ref:
    for key in ['indices','labels']:np.testing.assert_array_equal(z[key],ref[key])
    assert z['scores'].shape==(1024,) and np.isfinite(z['scores']).all()
    qfirst=int(ref['scores'][0].argmax());assert row['q0_first_local']==qfirst
    for name,order in orderings(z['scores'],qfirst).items():
     assert len(set(order))==1024
     k=first_success(order,z['labels']);assert k==row['first_success'][name];methods[name]['first_success'][task]=k
   out['verified_cases']+=1
  if not smoke:
   for name in ['q0','tta']:methods[name]=original['panels'][panel]['methods'][name]
   for name in ['uniform','uniform_centered']:methods[name]=centered['panels'][panel]['methods'][name]
  for name,x in methods.items():
   counts={f's{k}':sum(v is not None and v<=k for v in x['first_success'].values()) for k in [1,3,5,9]}
   if 'counts' in x:assert counts==x['counts']
   x['counts']=counts
  paired={}
  if not smoke:
   for name in ['q0','uniform_centered','tta']:
    for ref in ['base','base_q0_first']:
     paired[name+'_vs_'+ref]={str(k):{'gains':[t for t in tasks if methods[name]['first_success'][t] is not None and methods[name]['first_success'][t]<=k and not(methods[ref]['first_success'][t] is not None and methods[ref]['first_success'][t]<=k)],'losses':[t for t in tasks if methods[ref]['first_success'][t] is not None and methods[ref]['first_success'][t]<=k and not(methods[name]['first_success'][t] is not None and methods[name]['first_success'][t]<=k)]} for k in [1,3,5,9]}
  out['panels'][panel]={'methods':methods,'paired':paired}
 if smoke:out['passed']=True
 write(R/('SMOKE.json' if smoke else 'SUMMARY.json'),out)
 print(json.dumps({p:{m:x['counts'] for m,x in d['methods'].items()} for p,d in out['panels'].items()},indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['register','run','summary']);a.add_argument('--panel');a.add_argument('--shard',type=int,default=0);a.add_argument('--nshards',type=int,default=1);a.add_argument('--smoke',action='store_true');a=a.parse_args()
 if a.mode=='register':register()
 elif a.mode=='run':run(a.panel,a.shard,a.nshards,a.smoke)
 else:summary(a.smoke)
