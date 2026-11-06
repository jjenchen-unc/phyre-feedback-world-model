from transfer_core import *
from transfer_probe import update,cached
import argparse,numpy as np

def gated_scores(t,s0,s1,weights):
 import torch
 values=[]
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for start in range(0,1024,32):
   z,a=t.chunk(start);w=torch.from_numpy(np.asarray(weights[start:start+32],dtype=np.float32)).cuda()[:,None,None]
   st=s0+(s1-s0)*w;st=torch.where(w==1,s1,torch.where(w==0,s0,st))
   pred=t.model(z,a,st);values.append(t.scorer(pred,[f[None] for f in t.pool['features'][start:start+32]]).cpu().numpy())
 return np.concatenate(values)

def smoke(panel):
 import torch
 plan=checked();t=team(panel);task=read(L/'PLAN.json')['diagnostic_eval_tasks'][panel][0];t.load(task)
 try:
  with torch.no_grad():
   s0=t.model.initial_state().detach();first=t.scores(s0);j=int(first.argmax());assert t.pool['status'][j]<0;obs=cached(t,[j])[j]
   for mode in ['raw','centered']:
    s1=update(t,s0,j,obs,mode);scores=t.scores(s1)
    src=(B/'evaluation_128'/panel/'uniform' if mode=='raw' else C/'evaluation'/panel/'uniform')/task.replace(':','_')
    with np.load(src/'scores.npz') as z:np.testing.assert_array_equal(first,z['scores'][0]);np.testing.assert_array_equal(scores,z['scores'][1])
    np.testing.assert_array_equal(gated_scores(t,s0,s1,np.ones(1024)),scores);np.testing.assert_array_equal(gated_scores(t,s0,s1,np.zeros(1024)),first)
    selected=[j];v=scores.copy();v[j]=-np.inf;success=None
    for k in range(8):
     a=int(v.argmax());v[a]=-np.inf;selected.append(a)
     if t.pool['status'][a]>0:success=k+2;break
    ref=read(R/'freeze'/panel/mode/task.replace(':','_')/'result.json');assert selected==ref['selected'] and success==ref['first_success']
   raw=update(t,s0,j,obs,'raw');center=update(t,s0,j,obs,'centered');norm=update(t,s0,j,obs,'norm_matched');torch.testing.assert_close((norm-s0).norm(),(center-s0).norm(),rtol=1e-5,atol=1e-6)
  write(R/f'SMOKE_{panel}.json',{'passed':True,'plan_sha256':sha(R/'PLAN.json'),'freeze_trace_exact':True,'gate_zero_one_scores_exact':True,'norm_matched_verified':True,'task':task})
 finally:lr.close(t)

def run(panel,policy,shard,nshards):
 import torch
 plan=checked();assert read(R/f'SMOKE_{panel}.json')['passed'];decision=read(R/'ROUND2_DECISION.json')['panels'][panel]
 if policy!='norm_matched_full' and not decision['run_locality_gate']:
  write(R/'skipped'/panel/f'{policy}_{shard}.json',{'skipped':True,'reason':'Action-distance proxy direction not consistent across train/evaluation diagnostic scenes'});return
 t=team(panel);ck=read(B/'models'/panel/'uniform/checkpoint_128.json');model_hash=lr.source().old.ident(t.model)
 try:
  for task in plan['tasks'][panel][shard::nshards]:
   dest=R/'policies'/panel/policy/task.replace(':','_')
   if (dest/'result.json').exists():assert read(dest/'result.json')['plan_sha256']==sha(R/'PLAN.json');continue
   t.load(task);s0=t.model.initial_state().detach();state=s0;tried=np.zeros(1024,bool);scores_hist=[];selected=[];success=None;start=time.time();fixed=None
   with np.load(S/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as z:first=z['scores'][0].copy();np.testing.assert_array_equal(z['indices'],t.pool['action_idx']);np.testing.assert_array_equal(z['labels'],t.pool['status']>0)
   with torch.no_grad():
    for turn in range(9):
     scores=t.scores(state) if turn==0 or policy=='norm_matched_full' else fixed
     if turn==0:np.testing.assert_array_equal(scores,first)
     j=lr.source().pools.choose_trial(scores,tried,turn);tried[j]=True;selected.append(j);scores_hist.append(scores)
     if t.pool['status'][j]>0:success=turn+1;break
     if turn==8:break
     if policy=='norm_matched_full':
      obs=t.observations([j])[j];state=update(t,state,j,obs,'norm_matched')
     elif turn==0:
      obs=cached(t,[j])[j];state=update(t,s0,j,obs,'raw');std=np.maximum(t.pool['action'].std(0),1e-8);distance=np.sqrt((((t.pool['action']-t.pool['action'][j])/std)**2).sum(1));sigma=decision['action_sigma'];weights=np.exp(-distance**2/(2*sigma*sigma));eligible=np.flatnonzero(~tried)
      if policy=='uniform_gate_first':weights[eligible]=weights[eligible].mean()
      elif policy=='shuffled_gate_first':
       rng=np.random.default_rng(int(hashlib.sha256(('gate-shuffle-v1'+task).encode()).hexdigest()[:8],16));weights[eligible]=weights[eligible][rng.permutation(len(eligible))]
      else:assert policy=='action_gate_first'
      fixed=gated_scores(t,s0,state,weights)
   assert lr.source().old.ident(t.model)==model_hash
   lr.source().data.save_npz(dest/'scores.npz',scores=np.stack(scores_hist),selected=np.array(selected),indices=t.pool['action_idx'],labels=t.pool['status']>0)
   write(dest/'result.json',{'complete':True,'plan_sha256':sha(R/'PLAN.json'),'checkpoint_sha256':ck['checkpoint_sha256'],'task':task,'policy':policy,'first_success':success,'first_scores_exact':True,'model_unchanged':True,'seconds':time.time()-start});print('POLICY',panel,policy,task,success,flush=True)
 finally:lr.close(t)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--panel',required=True);a.add_argument('--policy');a.add_argument('--shard',type=int,default=0);a.add_argument('--nshards',type=int,default=8);a.add_argument('--smoke',action='store_true');a=a.parse_args();smoke(a.panel) if a.smoke else run(a.panel,a.policy,a.shard,a.nshards)
