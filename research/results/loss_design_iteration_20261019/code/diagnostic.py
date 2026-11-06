"""Fixed first-support/query diagnosis and training-only loss-scale calibration."""
from loss_runtime import *
import argparse,numpy as np

def inputs(team,task,case_index=None):
 import torch
 src=source();team.load(task);state=team.model.initial_state().detach();scores=team.scores(state);j=int(scores.argmax());assert team.pool['status'][j]<0
 if case_index is not None:_,_,perm,times=src.prior.draw_episode(team.plan['panels'][team.panel]['train_tasks'],case_index)
 else:
  rng=np.random.default_rng(int(hashlib.sha256(task.encode()).hexdigest()[:8],16));perm=rng.permutation(1024);times=[sorted(rng.choice(np.arange(1,32),8,replace=False).tolist()) for _ in range(8)]
 qi=[int(k) for k in perm if int(k)!=j][:8];obs=team.observations([j]+qi)
 return state,scores,j,qi,obs,times

def predict_query(team,state,qi):
 import torch
 with torch.autocast('cuda',dtype=torch.bfloat16):return team.model(torch.from_numpy(team.pool['z0'][qi].copy()).cuda().float(),torch.from_numpy(team.pool['action'][qi]).cuda(),state.expand(len(qi),-1,-1)).float()

def error_metrics(pred,target):
 e=pred[:,1:]-target[:,1:];mu=e.mean((1,2),keepdim=True);sq=e.square();flat=sq.flatten();threshold=flat.quantile(.99)
 return {'l1':float(e.abs().mean()),'mse':float(sq.mean()),'mean_bias_mse':float(mu.square().mean()),'centered_mse':float((e-mu).square().mean()),'abs_median':float(e.abs().median()),'largest1pct_energy_share':float(flat[flat>=threshold].sum()/flat.sum())}

def ranking(scores,status,first):
 available=np.arange(len(scores))!=first;y=status[available]>0;v=scores[available];ids=np.flatnonzero(available);order=np.argsort(-v,kind='stable');hits=np.flatnonzero(y[order]);pos=v[y];neg=v[~y]
 return {'next_success':bool(y[order[0]]),'best_success_rank':int(hits[0]+1) if len(hits) else None,'top8_success':bool(y[order[:8]].any()),'auc':float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean()) if len(pos) and len(neg) else None,'next_action_local':int(ids[order[0]])}

def run(panel):
 import torch
 from memory_geometry_followup import geometry_loss
 plan=checked();src=source();team=make_team(panel);ck=load(team,panel);rows=[];calcases=[];start=time.time()
 try:
  for split,cases in [('train',plan['training_calibration_cases'][panel]),('evaluation',[{'task':t,'index':None} for t in plan['diagnostic_eval_tasks'][panel]])]:
   for case in cases:
    load(team,panel);team.input_mode='raw';state,first,j,qi,obs,times=inputs(team,case['task'],case['index']);target=torch.from_numpy(np.stack([obs[k]['latent'] for k in qi])).cuda().float()
    cases_out={};states={}
    with torch.no_grad():
     states['q0']=state
     states['raw']=team.update(state,j,obs[j]);team.input_mode='centered';states['raw_centered']=team.update(state,j,obs[j]);team.input_mode='raw'
     for method in ['q0','raw','raw_centered']:
      pred=predict_query(team,states[method],qi);scores=team.scores(states[method]);m=error_metrics(pred,target);m['geometry']=float(geometry_loss(team.probe,pred,[obs[k]['features'] for k in qi],times));m['update_norm']=float((states[method]-state).norm());m.update(ranking(scores,team.pool['status'],j));cases_out[method]=m
      src.data.save_npz(R/'diagnostic'/panel/split/case['task'].replace(':','_')/(method+'.npz'),scores=scores,query=np.array(qi),action_idx=team.pool['action_idx'],status=team.pool['status'])
     # New center-trained checkpoint: same Q0 first state/support/query.
     centerck=read(C/'models'/panel/'centered/complete.json');assert sha(centerck['checkpoint'])==centerck['checkpoint_sha256']
     team.model.load_state_dict(torch.load(centerck['checkpoint'],map_location='cpu',weights_only=False)['state_dict']);team.input_mode='centered';cs=team.update(state,j,obs[j]);pred=predict_query(team,cs,qi);scores=team.scores(cs);m=error_metrics(pred,target);m['geometry']=float(geometry_loss(team.probe,pred,[obs[k]['features'] for k in qi],times));m['update_norm']=float((cs-state).norm());m.update(ranking(scores,team.pool['status'],j));cases_out['center_trained']=m
     src.data.save_npz(R/'diagnostic'/panel/split/case['task'].replace(':','_')/'center_trained.npz',scores=scores,query=np.array(qi),action_idx=team.pool['action_idx'],status=team.pool['status'])
    row={'split':split,'task':case['task'],'support_action':int(team.pool['action_idx'][j]),'query_actions':team.pool['action_idx'][qi].tolist(),'query_positive_count':int((team.pool['status'][qi]>0).sum()),'methods':cases_out};rows.append(row)
    if split=='train':calcases.append({'case':case,'j':j,'qi':qi,'times':times,'abs_median':cases_out['raw']['abs_median']})
    write(R/f'DIAGNOSTIC_PROGRESS_{panel}.json',{'rows':rows,'seconds':time.time()-start});print('FIXED',panel,split,case['task'],flush=True)
    del target,states,cs,pred
  beta=float(np.median([x['abs_median'] for x in calcases]));assert beta>0
  records=[];load(team,panel);team.input_mode='raw';team.model.checkpoint_blocks=True
  # State gradient: same downstream frozen predictor, queries, output and target.
  for case in calcases:
   team.load(case['case']['task']);qi=case['qi'];obs=team.observations([case['j']]+qi)
   with torch.no_grad():state=team.update(team.model.initial_state(),case['j'],obs[case['j']]).detach()
   state.requires_grad_(True);pred=predict_query(team,state,qi);target=torch.from_numpy(np.stack([obs[k]['latent'] for k in qi])).cuda().float();terms=latent_terms(pred,target,beta);terms['geometry']=geometry_loss(team.probe,pred,[obs[k]['features'] for k in qi],case['times']);grads={k:torch.autograd.grad(v,state,retain_graph=True)[0].detach() for k,v in terms.items()};norms={k:float(g.norm()) for k,g in grads.items()};assert all(np.isfinite(v) and v>0 for v in norms.values())
   records.append({'task':case['case']['task'],'gradient_norm_wrt_final_state':norms,'cosine_vs_l1':{k:float(torch.nn.functional.cosine_similarity(g.flatten(),grads['l1'].flatten(),dim=0)) for k,g in grads.items()},'losses':{k:float(v) for k,v in terms.items()}})
   del state,pred,target,terms,grads
  scales={k:float(np.median([x['gradient_norm_wrt_final_state']['l1']/x['gradient_norm_wrt_final_state'][k] for x in records])) for k in ['l1','mse','smooth']}
  write(R/f'CALIBRATION_{panel}.json',{'complete':True,'plan_sha256':sha(R/'PLAN.json'),'beta':beta,'scales':scales,'records':records,'calibration_scope':'Training-only frozen raw128 final-state gradients, approximate scale match. Not full-parameter or whole-training equivalence.'})
  write(R/f'DIAGNOSTIC_{panel}.json',{'complete':True,'plan_sha256':sha(R/'PLAN.json'),'rows':rows,'seconds':time.time()-start});print('CALIBRATION',panel,beta,scales,flush=True)
 finally:close(team)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--panel',required=True);run(a.parse_args().panel)
