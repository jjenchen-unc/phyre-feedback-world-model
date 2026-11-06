from common import *
from diagnostic import inputs,predict_query,error_metrics,ranking
import argparse,numpy as np

def run(panel,kind):
 import torch
 from memory_geometry_followup import geometry_loss
 plan=checked();team=make_team(panel);ck=load(team,panel,kind);rows=[]
 try:
  for split,cases in [('train',plan['training_calibration_cases'][panel]),('evaluation',[{'task':t,'index':None} for t in plan['diagnostic_eval_tasks'][panel]])]:
   for case in cases:
    state,first,j,qi,obs,times=inputs(team,case['task'],case['index']);target=torch.from_numpy(np.stack([obs[k]['latent'] for k in qi])).cuda().float()
    with torch.no_grad():
     updated=team.update(state,j,obs[j]);pred=predict_query(team,updated,qi);scores=team.scores(updated);m=error_metrics(pred,target);m['geometry']=float(geometry_loss(team.probe,pred,[obs[k]['features'] for k in qi],times));m['update_norm']=float((updated-state).norm());m.update(ranking(scores,team.pool['status'],j))
    reference=next(x for x in read(R/f'DIAGNOSTIC_{panel}.json')['rows'] if x['split']==split and x['task']==case['task']);assert reference['support_action']==int(team.pool['action_idx'][j]) and reference['query_actions']==team.pool['action_idx'][qi].tolist()
    rows.append({'task':case['task'],'split':split,'metrics':m});source().data.save_npz(R/'diagnostic'/panel/split/case['task'].replace(':','_')/(kind+'.npz'),scores=scores,query=np.array(qi),action_idx=team.pool['action_idx'],status=team.pool['status']);print('FIXED_NEW',panel,kind,split,case['task'],flush=True)
  write(R/f'DIAGNOSTIC_{panel}_{kind}.json',{'complete':True,'rows':rows,'checkpoint_sha256':ck['checkpoint_sha256'],'plan_sha256':sha(R/'PLAN.json')})
 finally:close(team)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--panel',required=True);a.add_argument('--kind',required=True);a=a.parse_args();run(a.panel,a.kind)
