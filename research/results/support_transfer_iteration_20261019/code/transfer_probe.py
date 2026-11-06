"""Per-query correction identity, oracle residual alignment and deployable proxies."""
from transfer_core import *
import argparse,numpy as np

def cached(team,indices):
 for j in indices:
  key=f'{team.task.replace(":","_")}_{int(team.pool["action_idx"][j]):06d}';bucket=str(int(hashlib.sha256(key.encode()).hexdigest(),16)%32)
  assert any((root/bucket/(key+'.npz')).exists() for root in [team.truth.root,*team.truth.parents]),('Unplanned cachemiss',team.task,j)
 return team.observations(indices)

def update(team,state,j,obs,mode):
 import torch
 pred=team.predict(state,j)[j%32:j%32+1];res=torch.from_numpy(obs['latent'][None]).cuda().float()-pred
 def one(input_mode):
  team.input_mode=input_mode
  with torch.autocast('cuda',dtype=torch.bfloat16):return team.model.update_state(state,res,torch.from_numpy(team.pool['action'][j:j+1]).cuda())
 if mode=='raw':return one('raw')
 if mode=='centered':return one('centered')
 assert mode=='norm_matched';raw=one('raw');center=one('centered');dr=raw-state;dc=center-state;norm=dr.norm()
 if float(norm)==0:assert float(dc.norm())==0;return state
 return state+dr*(dc.norm()/norm)

def query_prediction(team,state,qi):
 import torch
 with torch.autocast('cuda',dtype=torch.bfloat16):return team.model(torch.from_numpy(team.pool['z0'][qi].copy()).cuda().float(),torch.from_numpy(team.pool['action'][qi]).cuda(),state.expand(len(qi),-1,-1)).float()

def ranks(x):
 order=np.argsort(x,kind='stable');out=np.empty(len(x),float);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and x[order[j]]==x[order[i]]:j+=1
  out[order[i:j]]=(i+j-1)/2;i=j
 return out

def rho(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float)
 if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
 return float(np.corrcoef(ranks(x),ranks(y))[0,1])

def first(panel,shard,nshards):
 import torch
 from memory_geometry_followup import geometry_loss
 from diagnostic import ranking
 plan=checked();t=team(panel);ref=read(L/f'DIAGNOSTIC_{panel}.json');tasks=ref['rows'][shard::nshards]
 try:
  for base in tasks:
   task=base['task'];split=base['split'];t.load(task);ids={int(v):i for i,v in enumerate(t.pool['action_idx'])};j=ids[base['support_action']];qi=[ids[x] for x in base['query_actions']];obs=cached(t,[j]+qi);target=torch.from_numpy(np.stack([obs[k]['latent'] for k in qi])).cuda().float();s0=t.model.initial_state().detach();methods={};query=[]
   with torch.no_grad():
    pred0=query_prediction(t,s0,qi);support_pred=t.predict(s0,j)[j%32:j%32+1];rs=torch.from_numpy(obs[j]['latent'][None]).cuda().float()-support_pred;e=target[:,1:]-pred0[:,1:]
    es=e.flatten(1);rss=rs[:,1:].flatten();cos=torch.nn.functional.cosine_similarity(es,rss[None],dim=1)
    sd=np.maximum(t.pool['action'].std(0),1e-8);dist=np.sqrt((((t.pool['action'][qi]-t.pool['action'][j])/sd)**2).sum(1));motion=(pred0[:,1:]-pred0[:,:1])-(support_pred[:,1:]-support_pred[:,:1]);pd=motion.square().mean((1,2,3))
    for mode in ['raw','centered','norm_matched']:
     st=update(t,s0,j,obs[j],mode);pred=query_prediction(t,st,qi);d=pred[:,1:]-pred0[:,1:];dot=(e*d).mean((1,2,3));power=d.square().mean((1,2,3));mse0=e.square().mean((1,2,3));mse1=(target[:,1:]-pred[:,1:]).square().mean((1,2,3));gain=mse0-mse1
     torch.testing.assert_close(gain,2*dot-power,rtol=1e-3,atol=1e-7)
     l1=(target[:,1:]-pred[:,1:]).abs().mean((1,2,3));geo=[float(geometry_loss(t.probe,pred[k:k+1],[obs[q]['features']],[list(range(1,32,4))])) for k,q in enumerate(qi)]
     if mode in ['raw','centered']:
      key='raw' if mode=='raw' else 'raw_centered'
      np.testing.assert_allclose(float(l1.mean()),base['methods'][key]['l1'],rtol=1e-6,atol=1e-7)
      with np.load(L/'diagnostic'/panel/split/task.replace(':','_')/(key+'.npz')) as z:scores=z['scores'].copy()
     else:scores=t.scores(st)
     src=lr.source();src.data.save_npz(R/'first'/panel/split/task.replace(':','_')/(mode+'.npz'),scores=scores,indices=t.pool['action_idx'],labels=t.pool['status']>0)
     methods[mode]={'update_norm':float((st-s0).norm()),'ranking':ranking(scores,t.pool['status'],j),'queries':[]}
     for k,q in enumerate(qi):
      dp=float(dot[k]);pw=float(power[k]);g=float(gain[k]);alpha=max(0.,min(1.,dp/pw)) if pw>0 else 0.
      methods[mode]['queries'].append({'action_idx':int(t.pool['action_idx'][q]),'solved':bool(t.pool['status'][q]>0),'l1':float(l1[k]),'mse_before':float(mse0[k]),'mse_after':float(mse1[k]),'mse_gain':g,'error_dot_correction':dp,'correction_power':pw,'geometry':geo[k],'harm_type':'wrong_direction' if g<0 and dp<=0 else 'overshoot' if g<0 else 'not_harmful','oracle_alpha':alpha,'oracle_mse_gain':2*alpha*dp-alpha*alpha*pw})
    for k,q in enumerate(qi):query.append({'action_idx':int(t.pool['action_idx'][q]),'support_query_residual_cosine':float(cos[k]),'action_distance':float(dist[k]),'predicted_motion_distance':float(pd[k])})
   result={'complete':True,'plan_sha256':sha(R/'PLAN.json'),'panel':panel,'task':task,'split':split,'support_action':base['support_action'],'queries':query,'methods':methods,'oracle_limit':'Query residual and oraclealpha use querytruth for diagnosis only; never select deploymentactions with them.'}
   write(R/'first'/panel/split/task.replace(':','_')/'result.json',result);print('TRANSFER_FIRST',panel,split,task,flush=True)
 finally:lr.close(t)

def history(panel,shard,nshards):
 import torch
 from diagnostic import ranking,error_metrics
 from memory_geometry_followup import geometry_loss
 plan=checked();t=team(panel);refs=[x for x in read(L/f'DIAGNOSTIC_{panel}.json')['rows'] if x['split']=='evaluation'][shard::nshards]
 try:
  for ref in refs:
   task=ref['task'];t.load(task);idmap={int(a):j for j,a in enumerate(t.pool['action_idx'])};histories={};score_refs={}
   for origin,root in [('raw',B/'evaluation_128'/panel/'uniform'),('centered',C/'evaluation'/panel/'uniform')]:
    with np.load(root/task.replace(':','_')/'scores.npz') as z:score_refs[origin]=z['scores'].copy();selected=z['selected'].tolist();labels=z['labels'].copy()
    supports=[]
    for j in selected[:-1]:
     assert not labels[j];supports.append(j)
    histories[origin]=supports
   depth=min(4,*[len(x) for x in histories.values()]);assert depth>=1
   union={j for h in histories.values() for j in h[:depth]};qi=[idmap[a] for a in ref['query_actions'] if idmap[a] not in union];assert qi
   obs=cached(t,sorted(union|set(qi)));target=torch.from_numpy(np.stack([obs[j]['latent'] for j in qi])).cuda().float();records=[]
   with torch.no_grad():
    for origin,supports in histories.items():
     for mode in ['raw','centered','norm_matched']:
      state=t.model.initial_state().detach();norms=[]
      for step,j in enumerate(supports[:depth]):
       previous=state;state=update(t,state,j,obs[j],mode);norms.append(float((state-previous).norm()))
       if step not in {0,depth-1}:continue
       scores=t.scores(state)
       if mode==origin:np.testing.assert_array_equal(scores,score_refs[origin][step+1])
       pred=query_prediction(t,state,qi);m=error_metrics(pred,target);m['geometry']=float(geometry_loss(t.probe,pred,[obs[j]['features'] for j in qi],[list(range(1,32,4))]*len(qi)))
       # Mask all actually tried actions in this forced history, not only the first.
       status=t.pool['status'].copy();available=np.ones(1024,bool);available[supports[:step+1]]=False;v=scores[available];y=status[available]>0;order=np.argsort(-v,kind='stable');hits=np.flatnonzero(y[order]);pos=v[y];neg=v[~y]
       m.update(next_success=bool(y[order[0]]),top8_success=bool(y[order[:8]].any()),best_success_rank=int(hits[0]+1) if len(hits) else None,auc=float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean()) if len(pos) and len(neg) else None)
       records.append({'history':origin,'operator':mode,'updates':step+1,'metrics':m,'update_norms':norms.copy(),'query_count':len(qi),'matching_policy_scores_exact':mode==origin})
       lr.source().data.save_npz(R/'history'/panel/task.replace(':','_')/f'{origin}_{mode}_{step+1}.npz',scores=scores,tried=np.array(supports[:step+1]),query=np.array(qi),indices=t.pool['action_idx'],labels=t.pool['status']>0)
   write(R/'history'/panel/task.replace(':','_')/'result.json',{'complete':True,'task':task,'depth':depth,'records':records,'query_actions':t.pool['action_idx'][qi].tolist(),'limit':'Forced histories, not actual counterfactual policyrollouts. Residual recomputed fromeachoperator currentstate.'});print('TRANSFER_HISTORY',panel,task,depth,flush=True)
 finally:lr.close(t)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['first','history']);a.add_argument('--panel',required=True);a.add_argument('--shard',type=int,default=0);a.add_argument('--nshards',type=int,default=4);a=a.parse_args();(first if a.stage=='first' else history)(a.panel,a.shard,a.nshards)
