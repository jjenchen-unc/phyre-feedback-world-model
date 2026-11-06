"""Matched closed-loop evaluation of fresh query-composition training arms."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import train as tr
R=tr.R;S=tr.S;read=tr.read;write=tr.write;sha=tr.sha

def register(slot):
 plan=read(R/'PLAN.json');spec={'training_plan_sha256':sha(R/'PLAN.json'),'evaluation_code_sha256':sha(__file__),'slot':slot,'tasks':{p:s['tasks'] for p,s in plan['panels'].items()},'arms':plan['arms'],'budget':9,'controls':'sameQ0first, sameclassifier1024pool, samefrozenreader/probe/scorer; freshuniform and balanced trained for same slots'}
 path=R/f'EVAL_PLAN_{slot}.json'
 if path.exists():assert read(path)==spec
 else:write(path,spec)
 print('EVAL_REGISTERED',slot,flush=True)

def evaluate(panel,kind,slot,shard,nshards):
 import numpy as np,torch
 import selector_feedback_run as src
 from fast_inference import CandidateRanker
 torch.set_num_threads(2);assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
 spec=read(R/f'EVAL_PLAN_{slot}.json');assert spec['evaluation_code_sha256']==sha(__file__) and spec['training_plan_sha256']==sha(R/'PLAN.json')
 training=read(R/'PLAN.json');assert sha(R/'code/train.py')==training['code_sha256']
 paired=[read(R/'models'/panel/k/f'checkpoint_{slot}.json') for k in spec['arms']]
 assert all(x['slots']==slot and x['plan_sha256']==spec['training_plan_sha256'] for x in paired)
 assert len({x['initial_sha256'] for x in paired})==len({x['reader_sha256'] for x in paired})==1
 ck=read(R/'models'/panel/kind/f'checkpoint_{slot}.json');assert sha(ck['checkpoint'])==ck['checkpoint_sha256']
 team=tr.make_team(panel,kind,src.data.checked());team.model.load_state_dict(torch.load(ck['checkpoint'],map_location='cpu',weights_only=False)['state_dict']);team.model.eval().requires_grad_(False)
 assert src.old.reader_hash(team.model)==ck['reader_sha256'];fixed=src.old.ident(team.model);prov={'evaluation_plan_sha256':sha(R/f'EVAL_PLAN_{slot}.json'),'checkpoint_sha256':ck['checkpoint_sha256']}
 try:
  for task in spec['tasks'][panel][shard::nshards]:
   dest=R/f'evaluation_{slot}'/panel/kind/task.replace(':','_')
   if (dest/'result.json').exists():assert read(dest/'result.json')['provenance']==prov;continue
   team.load(task);pool=team.pool;ranker=CandidateRanker(team.model,team.head,team.rule,torch.from_numpy(pool['z0'].copy()).cuda().float(),torch.from_numpy(pool['action']).cuda(),[f[None] for f in pool['features']])
   state=team.model.initial_state().detach().clone();tried=np.zeros(1024,bool);events=[];history=[];selected=[];success=None;began=time.time()
   with np.load(S/'evaluation'/panel/'residual'/task.replace(':','_')/'scores.npz') as z:
    first=z['scores'][0].copy();np.testing.assert_array_equal(z['indices'],pool['action_idx']);np.testing.assert_array_equal(z['labels'],pool['status']>0)
   with torch.no_grad():
    for turn in range(9):
     ranked,rank_s=src.timed(lambda:ranker.rank(state,tried));values=ranked.scores
     if turn==0:np.testing.assert_array_equal(values,first)
     j=src.pools.choose_trial(values,tried,turn);assert not tried[j];tried[j]=True;selected.append(j);history.append(values)
     e={'turn':turn+1,'local':j,'action_idx':int(pool['action_idx'][j]),'solved':bool(pool['status'][j]>0),'rank_s':rank_s,'update_s':0.,'observation_s':0.};events.append(e)
     if e['solved']:success=turn+1;break
     if turn==8:break
     obs,e['observation_s']=src.timed(lambda:team.observations([j])[j]);e.update(cache_hit=obs['cache_hit'],observation_encode_s=obs['observation_encode_s'])
     state,e['update_s']=src.timed(lambda:team.update(state,j,obs))
   assert src.old.ident(team.model)==fixed
   src.data.save_npz(dest/'scores.npz',scores=np.stack(history),selected=np.array(selected),indices=pool['action_idx'],labels=pool['status']>0)
   write(dest/'result.json',{'complete':True,'provenance':prov,'panel':panel,'kind':kind,'task':task,'first_success':success,'events':events,'first_scores_exact':True,'model_unchanged':True,'seconds':time.time()-began})
   print('EVAL',slot,panel,kind,task,success,round(time.time()-began,1),flush=True)
 finally:team.truth.close()
 write(R/f'evaluation_{slot}'/panel/kind/f'shard_{shard}.json',{'complete':True,'provenance':prov})

def summary(slot):
 import numpy as np
 spec=read(R/f'EVAL_PLAN_{slot}.json');ph=sha(R/f'EVAL_PLAN_{slot}.json');out={'complete':True,'slot':slot,'evaluation_plan_sha256':ph,'scope':'Matched fresh updaters with uniform vs balanced8queries; one initialization, half original training slots if128','panels':{}}
 for panel,tasks in spec['tasks'].items():
  first={};training={}
  for kind in spec['arms']:
   ck=read(R/'models'/panel/kind/f'checkpoint_{slot}.json');training[kind]=ck;first[kind]={}
   for task in tasks:
    folder=R/f'evaluation_{slot}'/panel/kind/task.replace(':','_');row=read(folder/'result.json');assert row['complete'] and row['provenance']=={'evaluation_plan_sha256':ph,'checkpoint_sha256':ck['checkpoint_sha256']} and row['first_scores_exact'] and row['model_unchanged']
    with np.load(folder/'scores.npz') as z,np.load(S/'evaluation'/panel/'residual'/task.replace(':','_')/'scores.npz') as ref:
     np.testing.assert_array_equal(z['indices'],ref['indices']);np.testing.assert_array_equal(z['labels'],ref['labels']);np.testing.assert_array_equal(z['scores'][0],ref['scores'][0]);selected=z['selected'];assert len(set(selected))==len(selected)
     for t,j in enumerate(selected):
      scores=z['scores'][t].copy();scores[selected[:t]]=-np.inf;assert int(scores.argmax())==j;assert row['events'][t]['local']==int(j) and row['events'][t]['solved']==bool(z['labels'][j])
     hits=np.flatnonzero(z['labels'][selected]);actual=int(hits[0]+1) if len(hits) else None;assert row['first_success']==actual and len(selected)==(actual or 9)
    first[kind][task]=actual
  for kind in ['q0','residual']:
   first['original_'+kind]={t:read(S/'evaluation'/panel/kind/t.replace(':','_')/'result.json')['first_success'] for t in tasks}
  methods={kind:{'n':len(tasks),'counts':{f's{k}':sum(v is not None and v<=k for v in fs.values()) for k in [1,3,5,9]},'first_success':fs} for kind,fs in first.items()}
  pair={'gains':[t for t in tasks if first['balanced'][t] is not None and first['uniform'][t] is None],'losses':[t for t in tasks if first['balanced'][t] is None and first['uniform'][t] is not None]}
  out['panels'][panel]={'methods':methods,'balanced_vs_uniform':pair,'training':training}
 write(R/f'SUMMARY_{slot}.json',out);print(json.dumps({p:{'counts':{m:x['counts'] for m,x in d['methods'].items()},'gains':len(d['balanced_vs_uniform']['gains']),'losses':len(d['balanced_vs_uniform']['losses'])} for p,d in out['panels'].items()},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','eval','summary']);p.add_argument('--panel');p.add_argument('--kind');p.add_argument('--slot',type=int,default=128);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=1);a=p.parse_args()
 if a.mode=='register':register(a.slot)
 elif a.mode=='summary':summary(a.slot)
 else:evaluate(a.panel,a.kind,a.slot,a.shard,a.nshards)
