"""Predictor-only AdaJEPA-style adaptation for independent PHYRE placement trials."""
import argparse,copy,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch

R=Path(__file__).resolve().parents[1];P=R.parents[1]
MEM=P/'results/small_official_20261010'
sys.path.insert(0,str(MEM/'code'))
sys.path.insert(0,str(MEM/'recovery_20261010'))
import recovery_runner as m
f=m.f;p=m.p;read=m.read;write=m.write;sha=m.sha
PACKS={'cross':m.DATA/'packs/cross_f1_dev_shortlist1024_pca128','within':m.DATA/'packs/within_f0_dev_shortlist512_pca128'}
SOURCES={'paper':'https://arxiv.org/abs/2606.32026v1','code':'https://github.com/agentic-learning-ai-lab/adajepa/blob/51d8665b7978824bd218decab9e05ddb6eb1f47b/planning/adajepa.py'}

def fingerprint(model,trainable=None):
 h=hashlib.sha256()
 for n,v in model.named_parameters():
  if trainable is None or v.requires_grad==trainable:h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

class Adapter:
 def __init__(self,model,lr,steps,replay=5):
  self.model=model;self.lr=lr;self.steps=steps;self.replay=replay
  model.eval().requires_grad_(False)
  for sub in [model.blocks[-1],model.out[0]]:sub.requires_grad_(True)
  self.params=[v for v in model.parameters() if v.requires_grad]
  self.names=[n for n,v in model.named_parameters() if v.requires_grad]
  self.snapshot=[v.detach().clone() for v in self.params];self.buffer=[]
  self.frozen=fingerprint(model,False)
 @torch.no_grad()
 def reset(self):
  for v,s in zip(self.params,self.snapshot):v.copy_(s);v.grad=None
  self.buffer=[];self.model.eval()
 def update(self,index,z0,action,truth):
  assert truth.shape==(1,32,196,128) and not truth.requires_grad
  self.buffer.append((int(index),z0.detach(),action.detach(),truth.detach()))
  self.buffer=self.buffer[-self.replay:]
  # Official trainer recreates Adam at each finetune call; weights persist across trials.
  optimizer=torch.optim.Adam(self.params,lr=self.lr)
  z=torch.cat([x[1] for x in self.buffer]);a=torch.cat([x[2] for x in self.buffer]);target=torch.cat([x[3] for x in self.buffer])
  losses=[];norms=[];torch.cuda.synchronize();start=time.perf_counter()
  with torch.enable_grad():
   for _ in range(self.steps):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.model(z,a).float()
    # Same full-trajectory L1 objective as this pretrained predictor; no cross-reset concatenation.
    loss=(pred[:,1:]-target[:,1:]).abs().mean();assert torch.isfinite(loss)
    loss.backward();norm=torch.linalg.vector_norm(torch.stack([v.grad.detach().norm() for v in self.params if v.grad is not None]))
    assert torch.isfinite(norm) and norm>0
    assert all(v.grad is None for v in self.model.parameters() if not v.requires_grad)
    optimizer.step();losses.append(float(loss.detach()));norms.append(float(norm))
  torch.cuda.synchronize();seconds=time.perf_counter()-start
  self.model.eval()
  return {'replay_indices':[x[0] for x in self.buffer],'losses':losses,'grad_norms':norms,'seconds':seconds}

def prepare():
 assert not (R/'PLAN.json').exists();parent=m.checked();sel=read(MEM/'selection.json');cal={}
 for panel,path in PACKS.items():
  with np.load(str(path)+'_meta.npz') as z:tasks=z['tasks']
  available=sorted(set(map(str,tasks)));per=5 if panel=='cross' else 1
  selected=sorted(t for tp in sorted({t[:5] for t in available}) for t in sorted([t for t in available if t[:5]==tp],key=lambda t:f.stable(('ada-cal',panel,t)))[:per])
  assert set(selected)<=set(sel['panels'][panel]['dev']) and not set(selected)&set(sel['panels'][panel]['selected'])
  assert len(selected)==(20 if panel=='cross' else 25)
  cal[panel]={'pack':str(path),'meta_sha256':sha(str(path)+'_meta.npz'),'tasks':selected,'candidates_per_task':1024 if panel=='cross' else 512}
 configs=[{'name':f'lr{lr:g}_u{u}','lr':lr,'steps':u,'replay':5} for lr in [3e-5,3e-4,1e-3] for u in [1,3]]
 plan={'code_sha256':sha(__file__),'memory_plan_sha256':sha(MEM/'PLAN.json'),'recovery_runtime_sha256':sha(MEM/'recovery_20261010/recovery_runner.py'),
  'sources':SOURCES,'calibration':cal,'configs':configs,'selection':'largest calibration S9, then S5, then S1; ties prefer fewer steps then lower lr',
  'adapted':'last transformer block and final LayerNorm; output linear, earlier blocks, action embedding, encoder/PCA and scorer frozen',
  'loss':'mean L1 on observed frames1..31; same as original base pretraining; equal weight per independent attempt',
  'optimizer':'Adam default betas/eps, no weight decay, recreated at every adaptation call as official trainer',
  'reset':'weights and buffer reset at each new scene, never between same-scene trials',
  'test':{'tasks_per_panel':100,'panels':['cross','within'],'budgets':[1,3,5,9,20],'primary':'S@9','pool':'exact corrected memory-run1024 shortlist and actual outcome labels'},
  'replication':'one deterministic run per panel; original base checkpoint shared by three memory seeds, no stochastic replay or duplicate seed claims',
  'scope':'AdaJEPA-style predictor-only PHYRE port, not full encoder+predictor or original MPC benchmark reproduction',
  'budget_comparison':'same interaction budget; separately report adaptation/ranking/observation seconds, no claim of matched wall clock or offline training'}
 write(R/'PLAN.json',plan);print('PREPARED', {k:len(v['tasks']) for k,v in cal.items()},flush=True)

def checked():
 d=read(R/'PLAN.json');assert d['code_sha256']==sha(__file__)
 assert d['memory_plan_sha256']==sha(MEM/'PLAN.json') and d['recovery_runtime_sha256']==sha(MEM/'recovery_20261010/recovery_runner.py')
 m.checked();return d

def first(scores,labels):return p.first_rank(scores,labels)

def audit(folder):
 r=read(folder/'result.json')
 with np.load(folder/'scores.npz') as z:
  labels=z['labels'];scores=z['scores'];selected=z['selected'];assert len(set(z['indices'].tolist()))==len(labels)
  assert first(z['frozen'],labels)==r['frozen_first_success']
  np.testing.assert_array_equal(scores[0],z['frozen'])
  tried=set();hit=None
  for i,(s,j) in enumerate(zip(scores,selected),1):
   assert np.isfinite(s).all();s=s.copy();s[list(tried)]=-np.inf
   assert int(np.argmax(s))==int(j) and int(j) not in tried;tried.add(int(j))
   if labels[j]:assert i==len(selected);hit=i
  assert hit==r['first_success'] and len(selected)==r['attempts']<=r['budget']
  assert len(r['updates'])==r['attempts']-1
  for i,event in enumerate(r['updates']):
   ids=selected[max(0,i-4):i+1].tolist();assert event['replay_indices']==ids and not labels[ids].any()
 return r

def case(folder,task,adapter,score,d,status,observe,budget,prov):
 if (folder/'result.json').exists():
  r=audit(folder);assert r['provenance']==prov;return r
 folder.mkdir(parents=True,exist_ok=True);adapter.reset()
 z0=torch.from_numpy(d['z0']).cuda().float();actions=torch.from_numpy(d['action']).cuda();features=[v[None] for v in d['features']]
 tried=np.zeros(len(actions),bool);labels=status>0;snapshots=[];selected=[];updates=[];success=None
 ranking_seconds=0.;observation_seconds=0.;torch.cuda.reset_peak_memory_stats()
 for attempt in range(1,budget+1):
  torch.cuda.synchronize();start=time.perf_counter()
  scores=p.static_scores(adapter.model,z0,actions,features,score)
  torch.cuda.synchronize();ranking_seconds+=time.perf_counter()-start
  assert np.isfinite(scores).all();masked=scores.copy();masked[tried]=-np.inf;chosen=int(np.argmax(masked))
  assert not tried[chosen];tried[chosen]=True;snapshots.append(scores);selected.append(chosen)
  if labels[chosen]:success=attempt;break
  if attempt==budget:break
  start=time.perf_counter();truth=observe(chosen);torch.cuda.synchronize();observation_seconds+=time.perf_counter()-start
  updates.append(adapter.update(chosen,z0[chosen:chosen+1],actions[chosen:chosen+1],truth))
 assert fingerprint(adapter.model,False)==adapter.frozen
 peak=torch.cuda.max_memory_allocated()/2**30
 adapter.reset();reset_scores=p.static_scores(adapter.model,z0,actions,features,score)
 np.testing.assert_array_equal(reset_scores,snapshots[0])
 f.save_npz(folder/'scores.npz',{'indices':d['action_idx'],'labels':labels,'scores':np.stack(snapshots),'selected':np.array(selected),'frozen':reset_scores})
 r={'task':task,'provenance':prov,'first_success':success,'frozen_first_success':first(reset_scores,labels),
  'budget':budget,'attempts':len(selected),'updates':updates,'ranking_seconds':ranking_seconds,'observation_seconds':observation_seconds,
  'adaptation_seconds':sum(x['seconds'] for x in updates),'peak_gpu_GiB':peak,'reset_scores_exact':True,'frozen_parameters_unchanged':True}
 write(folder/'result.json',r);audit(folder);print('CASE',task,success,'frozen',r['frozen_first_success'],flush=True);return r

def counts(rows):return {f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in rows) for k in [1,3,5,9,20]}

def calibration(panel):
 plan=checked();cfg=plan['calibration'][panel];assert sha(cfg['pack']+'_meta.npz')==cfg['meta_sha256']
 pack=f.Pack(cfg['pack']);base,head,rule,_=m.load_base(panel);score=p.score_function(head,rule);summaries=[]
 for config in plan['configs']:
  adapter=Adapter(base,config['lr'],config['steps']);rows=[]
  for task in cfg['tasks']:
   ids=pack.by[task];assert len(ids)==cfg['candidates_per_task']
   d={'z0':np.array(pack.lat[0][ids,0]),'action':pack.actions[ids],'features':np.stack([pack.feature(i)[0] for i in ids]),'action_idx':np.array(ids)}
   def observe(j):return torch.from_numpy(pack.z([ids[j]])).cuda().float()
   prov={'plan':sha(R/'PLAN.json'),'panel':panel,'config':config,'role':'calibration','meta':cfg['meta_sha256']}
   rows.append(case(R/'calibration'/panel/config['name']/task.replace(':','_'),task,adapter,score,d,pack.labels[ids].astype(int)*2-1,observe,9,prov))
  summaries.append({'config':config,'counts':counts(rows)});adapter.reset()
 selected=sorted(summaries,key=lambda x:(-x['counts']['s9'],-x['counts']['s5'],-x['counts']['s1'],x['config']['steps'],x['config']['lr']))[0]
 write(R/'calibration'/panel/'SELECTED.json',{'complete':True,'plan':sha(R/'PLAN.json'),'n':len(cfg['tasks']),'candidates':summaries,'selected':selected})
 print('SELECTED',panel,selected,flush=True)

def evaluate(panel):
 plan=checked();selected=read(R/'calibration'/panel/'SELECTED.json');assert selected['plan']==sha(R/'PLAN.json')
 config=selected['selected']['config'];base,head,rule,_=m.load_base(panel);adapter=Adapter(base,config['lr'],config['steps']);score=p.score_function(head,rule)
 obs=m.Observations();rows=[];folder=R/'evaluation'/panel
 for task in read(MEM/'selection.json')['panels'][panel]['selected']:
  d,status,path=m.input_case(panel,task)
  def observe(j):return obs.get(panel,task,int(d['action_idx'][j]),d['action'][j],d['z0'][j],status[j])
  prov={'plan':sha(R/'PLAN.json'),'panel':panel,'config':config,'role':'evaluation','shortlist':sha(path),'labels':sha(path.with_name(path.stem+'_labels.npy'))}
  rows.append(case(folder/task.replace(':','_'),task,adapter,score,d,status,observe,20,prov))
  write(folder/'progress.json',{'scenes':len(rows),'total':100})
 obs.close();write(folder/'complete.json',{'complete':True,'plan':sha(R/'PLAN.json'),'n':100,'counts':counts(rows),'config':config})

def smoke():
 plan=checked();base,head,rule,_=m.load_base('within');original=fingerprint(base)
 # Engineering depth test on three distinct, already observed training trajectories.
 pack=f.Pack(str(m.DATA/'packs/within_f0_train_s16r16_pca128'));ids=pack.by[sorted(pack.by)[0]][:3]
 adapter=Adapter(base,3e-4,3);events=[];before=[x.detach().clone() for x in adapter.params]
 for i in ids:
  z=torch.from_numpy(pack.z([i])).cuda().float();a=torch.from_numpy(pack.actions[[i]]).cuda()
  events.append(adapter.update(i,z[:,0],a,z))
 assert any(not torch.equal(x,y) for x,y in zip(before,adapter.params))
 assert events[-1]['replay_indices']==ids and fingerprint(base,False)==adapter.frozen
 adapter.reset();assert not adapter.buffer and fingerprint(base)==original
 del pack,base,adapter;gc.collect();torch.cuda.empty_cache()
 # Real closed-loop calibration task: selector sees only initial inputs, observe callback sees chosen index.
 panel='cross';cfg=plan['calibration'][panel];pack=f.Pack(cfg['pack']);base,head,rule,_=m.load_base(panel);adapter=Adapter(base,3e-4,1)
 task=cfg['tasks'][0];ids=pack.by[task];d={'z0':np.array(pack.lat[0][ids,0]),'action':pack.actions[ids],'features':np.stack([pack.feature(i)[0] for i in ids]),'action_idx':np.array(ids)}
 observe=lambda j:torch.from_numpy(pack.z([ids[j]])).cuda().float()
 r=case(R/'smoke/case',task,adapter,p.score_function(head,rule),d,pack.labels[ids].astype(int)*2-1,observe,3,{'smoke':True,'plan':sha(R/'PLAN.json')})
 write(R/'SMOKE.json',{'passed':True,'plan':sha(R/'PLAN.json'),'trainable_names':adapter.names,'depth3_gradients_and_reset':True,'events':events,'closed_loop':r})

def summary():
 checked();result={}
 for panel in ['cross','within']:
  tasks=read(MEM/'selection.json')['panels'][panel]['selected'];rows=[]
  for task in tasks:
   folder=R/'evaluation'/panel/task.replace(':','_');r=audit(folder);rows.append(r)
   d,status,path=m.input_case(panel,task)
   assert r['provenance']['shortlist']==sha(path) and r['provenance']['labels']==sha(path.with_name(path.stem+'_labels.npy'))
   with np.load(folder/'scores.npz') as a,np.load(MEM/'evaluation_fixed'/panel/'q0_s0'/task.replace(':','_')/'scores.npz') as b:
    np.testing.assert_array_equal(a['indices'],b['indices']);np.testing.assert_array_equal(a['labels'],b['labels']);np.testing.assert_array_equal(a['frozen'],b['original'])
  assert counts(rows)==read(R/'evaluation'/panel/'complete.json')['counts']
  result[panel]={'counts':counts(rows),'first_success':{r['task']:r['first_success'] for r in rows},'adaptation_seconds':sum(r['adaptation_seconds'] for r in rows),'updates':sum(len(r['updates']) for r in rows)}
 write(R/'SUMMARY.json',{'complete':True,'raw_audit_passed':True,'frozen_baseline_matches_memory_run_exactly':True,'plan':sha(R/'PLAN.json'),'splits':result})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command');ap.add_argument('--panel',choices=['cross','within'],default='cross');a=ap.parse_args();torch.set_num_threads(4)
 if a.command=='prepare':prepare()
 elif a.command=='smoke':smoke()
 elif a.command=='lane':
  assert read(R/'SMOKE.json')['passed'];calibration(a.panel);gc.collect();torch.cuda.empty_cache();evaluate(a.panel)
 elif a.command=='summary':summary()
 else:raise ValueError(a.command)
