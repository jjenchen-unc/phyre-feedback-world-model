from common import *
import argparse,numpy as np

def folder(panel,kind,seed,mode,task,smoke=False):return R/('smoke' if smoke else 'evaluation')/f'seed{seed}'/panel/kind/mode/task.replace(':','_')

def run(panel,kind,seed,mode,shard,nshards,smoke=False):
 import torch
 from fast_inference import CandidateRanker
 plan=checked();src=source();team=make_team(panel,seed,mode);ck=load(team,panel,kind,seed);team.model.eval().requires_grad_(False);fixed=src.old.ident(team.model)
 if not smoke:assert read(R/'SMOKE.json')['passed'] and read(R/'audits'/f'seed{seed}'/panel/f'{kind}.json')['complete']
 prov={'plan_sha256':sha(R/'PLAN.json'),'checkpoint_sha256':ck['checkpoint_sha256'],'kind':kind,'seed':seed,'input':mode,'smoke':smoke}
 tasks=[read(C/'EVAL_PLAN.json')['smoke_tasks'][panel]] if smoke else plan['panels'][panel]['tasks'][shard::nshards]
 try:
  for task in tasks:
   dest=folder(panel,kind,seed,mode,task,smoke)
   if (dest/'result.json').exists():assert read(dest/'result.json')['provenance']==prov;continue
   team.load(task);pool=team.pool;actions=torch.from_numpy(pool['action']).cuda();ranker=CandidateRanker(team.model,team.head,team.rule,torch.from_numpy(pool['z0'].copy()).cuda().float(),actions,[f[None] for f in pool['features']]);state=team.model.initial_state().detach().clone();tried=np.zeros(1024,bool);events=[];scores=[];selected=[];success=None;start=time.time()
   with np.load(S/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as z:first=z['scores'][0].copy();np.testing.assert_array_equal(z['indices'],pool['action_idx']);np.testing.assert_array_equal(z['labels'],pool['status']>0)
   if smoke:
    with np.load(B/'evaluation_128'/panel/'uniform'/task.replace(':','_')/'scores.npz') as z:expected=z['scores'].copy()
   with torch.no_grad():
    for turn in range(9):
     ranked,rs=src.timed(lambda:ranker.rank(state,tried));v=ranked.scores
     if turn==0:np.testing.assert_array_equal(v,first)
     if smoke:np.testing.assert_array_equal(v,expected[turn])
     j=src.pools.choose_trial(v,tried,turn);tried[j]=True;selected.append(j);scores.append(v);event={'turn':turn+1,'local':j,'action_idx':int(pool['action_idx'][j]),'solved':bool(pool['status'][j]>0),'rank_s':rs,'update_s':0.,'observation_s':0.};events.append(event)
     if event['solved']:success=turn+1;break
     if turn==8:break
     if mode=='zero':
      def zero():
       with torch.autocast('cuda',dtype=torch.bfloat16):return team.model.update_state(state,torch.zeros(1,32,196,128,device='cuda'),actions[j:j+1])
      state,event['update_s']=src.timed(zero)
     else:
      obs,event['observation_s']=src.timed(lambda:team.observations([j])[j]);state,event['update_s']=src.timed(lambda:team.update(state,j,obs))
   assert src.old.ident(team.model)==fixed
   if smoke:assert len(expected)==len(scores)
   if mode=='zero':assert team.truth.encoder is None and team.truth.worker is None
   src.data.save_npz(dest/'scores.npz',scores=np.stack(scores),selected=np.array(selected),indices=pool['action_idx'],labels=pool['status']>0)
   write(dest/'result.json',{'complete':True,'task':task,'events':events,'first_success':success,'provenance':prov,'model_unchanged':True,'first_scores_exact':True,'replay_exact':smoke,'seconds':time.time()-start});print('EVAL',panel,kind,seed,mode,task,success,flush=True)
 finally:close(team)

def collect(panel,kind,seed,mode,smoke=False):
 plan=checked();ck=read(B/'models'/panel/'uniform/checkpoint_128.json') if kind=='l1' and seed==0 else read(R/'models'/f'seed{seed}'/panel/kind/'complete.json');first={}
 tasks=[read(C/'EVAL_PLAN.json')['smoke_tasks'][panel]] if smoke else plan['panels'][panel]['tasks']
 for task in tasks:
  dest=folder(panel,kind,seed,mode,task,smoke);row=read(dest/'result.json');assert row['complete'] and row['model_unchanged'] and row['first_scores_exact'];assert row['provenance']=={'plan_sha256':sha(R/'PLAN.json'),'checkpoint_sha256':ck['checkpoint_sha256'],'kind':kind,'seed':seed,'input':mode,'smoke':smoke}
  with np.load(dest/'scores.npz') as z,np.load(S/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as ref:
   for k in ['indices','labels']:np.testing.assert_array_equal(z[k],ref[k])
   np.testing.assert_array_equal(z['scores'][0],ref['scores'][0]);sel=z['selected'];assert len(sel)==len(set(sel))==len(z['scores'])==len(row['events']) and np.isfinite(z['scores']).all()
   for t,j in enumerate(sel):
    v=z['scores'][t].copy();v[sel[:t]]=-np.inf;assert int(v.argmax())==j;e=row['events'][t];assert e['local']==int(j) and e['action_idx']==int(z['indices'][j]) and e['solved']==bool(z['labels'][j])
   hits=np.flatnonzero(z['labels'][sel]);actual=int(hits[0]+1) if len(hits) else None;assert actual==row['first_success'] and len(sel)==(actual or 9)
   if smoke:
    assert row['replay_exact']
    with np.load(B/'evaluation_128'/panel/'uniform'/task.replace(':','_')/'scores.npz') as old:np.testing.assert_array_equal(z['scores'],old['scores'])
  first[task]=actual
 return {'n':len(tasks),'counts':{f's{k}':sum(v is not None and v<=k for v in first.values()) for k in [1,3,5,9]},'first_success':first}

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--panel',required=True);a.add_argument('--kind',required=True);a.add_argument('--seed',type=int,default=0);a.add_argument('--input',default='raw');a.add_argument('--shard',type=int,default=0);a.add_argument('--nshards',type=int,default=1);a.add_argument('--smoke',action='store_true');a=a.parse_args();run(a.panel,a.kind,a.seed,a.input,a.shard,a.nshards,a.smoke)
