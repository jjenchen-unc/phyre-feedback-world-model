"""Raw/centered/zero inputs on centered-trained checkpoint; exact old-policy smoke."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import train as tr
P,R,S,B=tr.P,tr.R,tr.S,tr.B
read,write,sha=tr.read,tr.write,tr.sha
OLD_CENTER=P/'results/query_balance_centered_eval_20261019'
OLD_ZERO=P/'results/fresh_zero_feedback_eval_20261019'
MODES=['centered','raw','zero']
def register():
 training=read(R/'PLAN.json')
 spec={'code_sha256':sha(__file__),'training_plan_sha256':sha(R/'PLAN.json'),'modes':MODES,'tasks':{p:v['tasks'] for p,v in training['panels'].items()},'budget':9,'smoke_tasks':read(OLD_CENTER/'PLAN.json')['smoke_tasks'],'primary':'New centered-trained/centered-input vs raw-trained/raw-input; compare raw-trained/centered-input to distinguish training adaptation. Raw/zero new-checkpoint inputs are diagnostic controls.'}
 if (R/'EVAL_PLAN.json').exists():assert read(R/'EVAL_PLAN.json')==spec
 else:write(R/'EVAL_PLAN.json',spec)
 print('EVAL_REGISTERED')
def reference_folder(panel,mode,task):
 root=B/'evaluation_128'/panel/'uniform' if mode=='raw' else OLD_CENTER/'evaluation'/panel/'uniform' if mode=='centered' else OLD_ZERO/'evaluation'/panel
 return root/task.replace(':','_')
def checkpoint(panel,smoke):
 return read((B/'models'/panel/'uniform/checkpoint_128.json') if smoke else (R/'models'/panel/'centered/checkpoint_128.json'))
def run(panel,mode,shard,nshards,smoke=False):
 import torch
 import selector_feedback_run as src
 from fast_inference import CandidateRanker
 torch.set_num_threads(2);assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
 plan=read(R/'EVAL_PLAN.json');training=read(R/'PLAN.json')
 assert sha(__file__)==plan['code_sha256'] and sha(R/'PLAN.json')==plan['training_plan_sha256'] and sha(R/'code/train.py')==training['code_sha256']
 if not smoke:
  assert read(R/'SMOKE.json')['passed'] and read(R/f'TRAINING_AUDIT_{panel}.json')['complete']
 ck=checkpoint(panel,smoke);assert sha(ck['checkpoint'])==ck['checkpoint_sha256']
 team=tr.make_team(panel,'centered',src.data.checked(),mode=mode)
 team.model.load_state_dict(torch.load(ck['checkpoint'],map_location='cpu',weights_only=False)['state_dict']);team.model.eval().requires_grad_(False)
 assert src.old.reader_hash(team.model)==ck['reader_sha256']
 fixed=src.old.ident(team.model);prov={'evaluation_plan_sha256':sha(R/'EVAL_PLAN.json'),'checkpoint_sha256':ck['checkpoint_sha256'],'mode':mode,'smoke':smoke}
 tasks=[plan['smoke_tasks'][panel]] if smoke else plan['tasks'][panel][shard::nshards]
 try:
  for task in tasks:
   folder=R/('smoke' if smoke else 'evaluation')/panel/mode/task.replace(':','_')
   if (folder/'result.json').exists():assert read(folder/'result.json')['provenance']==prov;continue
   team.load(task);pool=team.pool;actions=torch.from_numpy(pool['action']).cuda()
   ranker=CandidateRanker(team.model,team.head,team.rule,torch.from_numpy(pool['z0'].copy()).cuda().float(),actions,[f[None] for f in pool['features']])
   with np.load(S/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as z:
    first=z['scores'][0].copy();np.testing.assert_array_equal(z['indices'],pool['action_idx']);np.testing.assert_array_equal(z['labels'],pool['status']>0)
   expected=None
   if smoke:
    with np.load(reference_folder(panel,mode,task)/'scores.npz') as z:expected=z['scores'].copy()
   state=team.model.initial_state().detach().clone();tried=np.zeros(1024,bool);events=[];selected=[];scores=[];success=None;began=time.time()
   with torch.no_grad():
    for turn in range(9):
     ranked,rank_s=src.timed(lambda:ranker.rank(state,tried));values=ranked.scores
     if turn==0:np.testing.assert_array_equal(values,first)
     if smoke:np.testing.assert_array_equal(values,expected[turn])
     j=src.pools.choose_trial(values,tried,turn);tried[j]=True;selected.append(j);scores.append(values)
     event={'turn':turn+1,'local':j,'action_idx':int(pool['action_idx'][j]),'solved':bool(pool['status'][j]>0),'rank_s':rank_s,'update_s':0.,'observation_s':0.};events.append(event)
     if event['solved']:success=turn+1;break
     if turn==8:break
     if mode=='zero':
      def update():
       with torch.autocast('cuda',dtype=torch.bfloat16):return team.model.update_state(state,torch.zeros(1,32,196,128,device='cuda'),actions[j:j+1])
      state,event['update_s']=src.timed(update)
     else:
      obs,event['observation_s']=src.timed(lambda:team.observations([j])[j]);event.update(cache_hit=obs['cache_hit'],observation_encode_s=obs['observation_encode_s'])
      state,event['update_s']=src.timed(lambda:team.update(state,j,obs))
   assert src.old.ident(team.model)==fixed
   if smoke:assert len(scores)==len(expected)
   if mode=='zero':assert team.truth.encoder is None and team.truth.worker is None
   src.data.save_npz(folder/'scores.npz',scores=np.stack(scores),selected=np.array(selected),indices=pool['action_idx'],labels=pool['status']>0)
   write(folder/'result.json',{'complete':True,'provenance':prov,'task':task,'panel':panel,'mode':mode,'events':events,'first_success':success,'model_unchanged':True,'first_scores_exact':True,'full_replay_exact':smoke,'seconds':time.time()-began})
   print('EVAL',panel,mode,task,success,round(time.time()-began,1),flush=True)
 finally:team.center_hook.remove();team.truth.close()
def summary(smoke=False):
 plan=read(R/'EVAL_PLAN.json');ph=sha(R/'EVAL_PLAN.json')
 assert sha(__file__)==plan['code_sha256']
 out={'complete':True,'verified_rollouts':0,'evaluation_plan_sha256':ph,'panels':{},'primary':plan['primary']}
 for panel,tasks in plan['tasks'].items():
  tasks=[plan['smoke_tasks'][panel]] if smoke else tasks;ck=checkpoint(panel,smoke);methods={}
  for mode in MODES:
   first={}
   for task in tasks:
    folder=R/('smoke' if smoke else 'evaluation')/panel/mode/task.replace(':','_');row=read(folder/'result.json')
    assert row['complete'] and row['first_scores_exact'] and row['model_unchanged']
    assert row['provenance']=={'evaluation_plan_sha256':ph,'checkpoint_sha256':ck['checkpoint_sha256'],'mode':mode,'smoke':smoke}
    with np.load(folder/'scores.npz') as z,np.load(S/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as ref:
     for key in ['indices','labels']:np.testing.assert_array_equal(z[key],ref[key])
     np.testing.assert_array_equal(z['scores'][0],ref['scores'][0]);sel=z['selected'];assert len(sel)==len(set(sel))==len(z['scores'])==len(row['events']) and np.isfinite(z['scores']).all()
     for turn,j in enumerate(sel):
      values=z['scores'][turn].copy();values[sel[:turn]]=-np.inf;assert int(values.argmax())==j
      e=row['events'][turn];assert e['local']==int(j) and e['action_idx']==int(z['indices'][j]) and e['solved']==bool(z['labels'][j])
     hits=np.flatnonzero(z['labels'][sel]);actual=int(hits[0]+1) if len(hits) else None;assert row['first_success']==actual and len(sel)==(actual or 9)
     if smoke:
      assert row['full_replay_exact']
      with np.load(reference_folder(panel,mode,task)/'scores.npz') as prev:np.testing.assert_array_equal(z['scores'],prev['scores']);np.testing.assert_array_equal(sel,prev['selected'])
    first[task]=actual;out['verified_rollouts']+=1
   methods['center_trained_'+mode]={'n':len(tasks),'first_success':first,'counts':{f's{k}':sum(v is not None and v<=k for v in first.values()) for k in [1,3,5,9]}}
  paired={}
  if not smoke:
   controls=read(OLD_CENTER/'SUMMARY.json')['panels'][panel]['methods'];z=read(OLD_ZERO/'SUMMARY.json')['panels'][panel]['methods'];base=read(P/'results/plain_base_comparison_20261019/SUMMARY.json')['panels'][panel]['methods']
   methods.update(raw_trained_raw=controls['uniform'],raw_trained_centered=controls['uniform_centered'],raw_trained_zero=z['zero_intervention'],q0=base['q0'],base=base['base'],tta=base['tta'])
   for candidate,ref in [('center_trained_centered','raw_trained_raw'),('center_trained_centered','raw_trained_centered'),('center_trained_centered','center_trained_raw'),('center_trained_centered','center_trained_zero'),('center_trained_centered','q0')]:
    af,bf=methods[candidate]['first_success'],methods[ref]['first_success'];assert set(af)==set(bf)
    paired[candidate+'_vs_'+ref]={str(k):{'gains':[t for t in tasks if af[t] is not None and af[t]<=k and not(bf[t] is not None and bf[t]<=k)],'losses':[t for t in tasks if bf[t] is not None and bf[t]<=k and not(af[t] is not None and af[t]<=k)]} for k in [1,3,5,9]}
  out['panels'][panel]={'methods':methods,'paired':paired}
 if smoke:out['passed']=True
 write(R/('SMOKE.json' if smoke else 'SUMMARY.json'),out);print(json.dumps({p:{m:x['counts'] for m,x in d['methods'].items()} for p,d in out['panels'].items()},indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['register','run','summary']);a.add_argument('--panel');a.add_argument('--input',choices=MODES);a.add_argument('--shard',type=int,default=0);a.add_argument('--nshards',type=int,default=1);a.add_argument('--smoke',action='store_true');a=a.parse_args()
 if a.mode=='register':register()
 elif a.mode=='run':run(a.panel,a.input,a.shard,a.nshards,a.smoke)
 else:summary(a.smoke)
