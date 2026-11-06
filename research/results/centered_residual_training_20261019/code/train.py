"""Fresh uniform-query updater; centered residual in collection and full BPTT."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[3];S=P/'results/q0_first_feedback_20261018';R=P/'results/centered_residual_training_20261019';B=P/'results/query_balance_training_20261019'
sys.path[:0]=[str(S/'code'),str(S/'sunflower_backend'),str(P/'pipeline')]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def sample_query(permutation,selected,status,kind):
 assert kind=='centered'
 return [int(j) for j in permutation if int(j) not in set(selected)][:8]

def transform(z,mode):
 if mode=='centered':return z-z.mean((1,2),keepdim=True)
 if mode=='zero':return z*0
 assert mode=='raw'
 return z

def register():
 source=read(S/'PLAN.json')
 plan={'source_plan_sha256':sha(S/'PLAN.json'),'code_sha256':sha(__file__),
  'reference_training_plan_sha256':sha(B/'PLAN.json'),'reference_training_code_sha256':sha(B/'code/train.py'),
  'slots':128,'episodes_per_slot':2,'query_count':8,'arms':['centered'],'panels':source['panels'],'budget':9,
  'initialization':'Fresh updater seed7919, identical to prior uniform128 initialization; frozen Q0 reader, initialtokens, probes and scorer. No continuation.',
  'only_change':'Center pooled residual per feature across time/spatial positions before residual_in. Same transform as previously evaluated inference intervention. Applied during online support selection and differentiable fullBPTT loss replay.',
  'queries':'Uniform8untried, original deterministic permutation; no label balancing or near threshold',
  'loss':'Unchanged terminal futurelatentL1 + .1geometry;2episodes/slot,8queries,8geometrytimes/query,fullBPTT',
  'optimizer':'AdamW3e-4,wd.05,OneCycle128,pct_start.03,clip1; original scheduler including inactive slots',
  'paired_invariants':'Same initialweights/reader,scene/cap/Q0first,activeepisodes andoptimizerstep pattern as prioruniform128; later actions and query exclusions may differ.',
  'evaluation':'Same100scenes/protocol,9attempts,classifier1024pool,Q0first. Evaluate centered(primary),raw,zero inputs on new checkpoint; compare all3 on existing rawtrained128 checkpoint.',
  'limits':['One seed/fold perprotocol','Evaluation raw/zero are inputinterventions, may beOOD','Rawtrained128 control reused, identical initialization and schedule verified','Difference changes supporthistory as model learns; currentprediction residual recomputed eachtime'],
  'reference_checkpoints':{p:read(B/'models'/p/'uniform/checkpoint_128.json') for p in source['panels']}}
 if (R/'PLAN.json').exists():assert read(R/'PLAN.json')==plan
 else:write(R/'PLAN.json',plan)
 print('REGISTERED_CENTERED_TRAINING',flush=True)

def make_team(panel,kind,source_plan,mode="centered"):
 import numpy as np,torch
 import selector_feedback_run as src
 from runtime import IsolatedEncoder
 src.data.EncoderProxy=IsolatedEncoder
 class OwnTruth(src.TimedTruthCache):
  def __init__(self,panel):
   super().__init__(panel);self.parents=[B/'observations'/panel,P/'results/query_balance_centered_eval_20261019/observations'/panel,self.root,P/'results/memory_overnight_policy_20261019/observations'/panel];self.root=R/'observations'/panel
  def get(self,task,local,pool):
   key=f'{task.replace(":","_")}_{int(pool["action_idx"][local]):06d}';bucket=str(int(hashlib.sha256(key.encode()).hexdigest(),16)%32)
   for root in self.parents:
    path=root/bucket/(key+'.npz')
    if not path.exists():continue
    with np.load(path) as z:
     assert str(z['plan'])==sha(src.data.SOURCE_R/'PLAN.json') and str(z['task'])==task and int(z['status'])==int(pool['status'][local])
     np.testing.assert_array_equal(z['action'],pool['simulation_action'][local])
     return {'local':local,'latent':z['latent'].copy(),'features':z['features'].copy(),'status':int(z['status']),'action_idx':int(pool['action_idx'][local]),'cache_hit':True,'observation_encode_s':0.,'total_observation_s':0.}
   return super().get(task,local,pool)
 class Team(src.Team):
  def collect(self,index,verify=False,cap_override=None):
   original=src.prior.query_indices
   src.prior.query_indices=lambda permutation,selected:sample_query(permutation,selected,self.pool['status'],kind)
   try:ep=super().collect(index,verify,cap_override)
   finally:src.prior.query_indices=original
   ep['query_positive_count']=int(sum(self.pool['status'][j]>0 for j in ep['query']))
   return ep
 team=Team(panel,'residual',source_plan);team.truth.close();team.truth=OwnTruth(panel)
 team.input_mode=mode
 team.center_hook=team.model.residual_in.register_forward_pre_hook(lambda module,args:(transform(args[0],team.input_mode),))
 return team

def train(panel,kind):
 import numpy as np,torch
 import selector_feedback_run as src
 torch.set_num_threads(2);assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
 plan=read(R/'PLAN.json');assert sha(__file__)==plan['code_sha256'] and sha(S/'PLAN.json')==plan['source_plan_sha256']
 assert sha(B/'PLAN.json')==plan['reference_training_plan_sha256'] and sha(B/'code/train.py')==plan['reference_training_code_sha256']
 assert read(R/'SMOKE.json')['passed'] and read(R/'ENCODER_VALIDATION.json')['passed']
 team=make_team(panel,kind,src.data.checked());model=team.model;initial=src.old.ident(model);reader=src.old.reader_hash(model);probe=src.old.ident(team.probe);head=src.old.ident(team.head)
 reference=read(B/'models'/panel/'uniform/preflight.json')
 assert initial==reference['initial_sha256'] and reader==reference['reader_sha256']
 folder=R/'models'/panel/kind;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():assert read(folder/'complete.json')['plan_sha256']==sha(R/'PLAN.json');team.truth.close();return
 assert not (folder/'episodes.jsonl').exists(),'Do not silently restart a partially trained arm'
 opt,sched=src.optimizer(model,plan['slots']);gradient_checked=False;began=time.time();updates=0
 try:
  # Exact forward/gradient comparison on a real active training episode, before any optimizer step.
  for index in range(16):
   ep=team.collect(index,verify=True,cap_override=1)
   if not ep['supports']:continue
   tests=[]
   for whole in [False,True]:
    model.zero_grad(set_to_none=True);model.checkpoint_blocks=not whole
    with src.controlled_backward('cpu',True):
     loss,_,_=team.loss(ep,whole,True);loss.backward()
    tests.append((float(loss),src.old.v.fingerprint((n,p.grad) for n,p in model.named_parameters() if p.grad is not None)))
    del loss
   assert tests[0]==tests[1];assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters());gradient_checked=True
   del ep;break
  assert gradient_checked and src.old.ident(model)==initial
  model.checkpoint_blocks=False;model.zero_grad(set_to_none=True)
  # Non-identity checkpoint exercises residual projection and multi-update gradients.
  fresh={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  refck=plan['reference_checkpoints'][panel];assert sha(refck['checkpoint'])==refck['checkpoint_sha256']
  model.load_state_dict(torch.load(refck['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
  multistep=False
  for index in range(32):
   ep=team.collect(index,verify=True,cap_override=2)
   if len(ep['supports'])<2:continue
   tests=[]
   for whole in [False,True]:
    model.zero_grad(set_to_none=True);model.checkpoint_blocks=not whole
    with src.controlled_backward('cpu',True):
     loss,_,_=team.loss(ep,whole,True);loss.backward()
    assert model.residual_in.weight.grad is not None and model.residual_in.weight.grad.abs().sum()>0
    tests.append((float(loss),src.old.v.fingerprint((n,p.grad) for n,p in model.named_parameters() if p.grad is not None)))
    del loss
   assert tests[0]==tests[1];multistep=True;del ep;break
  assert multistep
  model.load_state_dict(fresh);del fresh
  model.checkpoint_blocks=False;model.zero_grad(set_to_none=True)
  assert src.old.ident(model)==initial and src.old.reader_hash(model)==reader
  write(folder/'preflight.json',{'passed':True,'gradient_exact':True,'nonidentity_two_update_gradient_exact':True,'same_initialization_as_raw128':True,'initial_sha256':initial,'reader_sha256':reader,'plan_sha256':sha(R/'PLAN.json'),'seconds':time.time()-began})
  barrier_start=time.time()
  while True:
   files=[R/'models'/p/k/'preflight.json' for p in ['cross','within'] for k in plan['arms']]
   if all(p.exists() for p in files):break
   if time.time()-barrier_start>900:raise TimeoutError('All-arm preflight barrier exceeded15min; inspect failed arm')
   time.sleep(2)
  checks=[read(p) for p in files]
  assert all(x['passed'] and x['plan_sha256']==sha(R/'PLAN.json') for x in checks)
  for pp in ['cross','within']:
   pair=[read(R/'models'/pp/k/'preflight.json') for k in plan['arms']]
   assert len({x['initial_sha256'] for x in pair})==len({x['reader_sha256'] for x in pair})==1
  began=time.time()
  with (folder/'episodes.jsonl').open('w',buffering=1) as f:
   reference_rows=[json.loads(line) for line in (B/'models'/panel/'uniform/episodes.jsonl').read_text().splitlines()]
   for slot in range(plan['slots']):
    start=time.time();opt.zero_grad(set_to_none=True);active=False;events=[]
    for offset in range(2):
     ep=team.collect(2*slot+offset);loss,l1,geo=team.loss(ep,True,slot<3);assert torch.isfinite(loss)
     if loss.requires_grad:(loss/2).backward();active=True
     event=team.event(ep,loss,l1,geo);event['query_positive_count']=ep['query_positive_count'];events.append(event);del ep,loss
    prior_events=reference_rows[slot]['events']
    assert active==reference_rows[slot]['optimizer_step']
    for event,refevent in zip(events,prior_events):
     assert (event['task'],event['cap'],event['selected'][0],bool(event['supports']))==(refevent['task'],refevent['cap'],refevent['selected'][0],bool(refevent['supports']))
    if active:
     norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
     assert torch.isfinite(norm) and norm>0;assert all(p.grad is None for p in model.parameters() if not p.requires_grad);opt.step();updates+=1
    sched.step();done=slot+1
    f.write(json.dumps({'slot':slot,'optimizer_step':active,'events':events,'seconds':time.time()-start})+'\n')
    if done%4==0:
     assert src.old.reader_hash(model)==reader and src.old.ident(team.probe)==probe and src.old.ident(team.head)==head
     progress={'slots':done,'total':plan['slots'],'optimizer_steps':updates,'elapsed_s':time.time()-began,'initial_sha256':initial,'reader_sha256':reader,'plan_sha256':sha(R/'PLAN.json')}
     write(folder/'progress.json',progress);print('TRAIN',panel,kind,done,round(progress['elapsed_s'],1),flush=True)
    if done in [32,64,96,128]:
     f.flush();os.fsync(f.fileno());path=folder/f'model_{done}.pt';src.old.b.atomic_save(path,{'state_dict':model.state_dict(),'plan_sha256':sha(R/'PLAN.json'),'slots':done})
     receipt={'complete':True,'slots':done,'checkpoint':str(path),'checkpoint_sha256':sha(path),'initial_sha256':initial,'reader_sha256':reader,'plan_sha256':sha(R/'PLAN.json'),'optimizer_steps':updates,'elapsed_s':time.time()-began}
     write(folder/f'checkpoint_{done}.json',receipt)
     if done==plan['slots']:write(folder/'complete.json',receipt)
 finally:team.center_hook.remove();team.truth.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','train']);p.add_argument('--panel',choices=['cross','within']);p.add_argument('--kind',choices=['centered']);a=p.parse_args();register() if a.mode=='register' else train(a.panel,a.kind)
