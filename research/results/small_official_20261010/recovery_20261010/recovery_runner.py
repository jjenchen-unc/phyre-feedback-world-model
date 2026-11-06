"""Small official-fold validation. All ranking inputs are initial observations only."""
import argparse,copy,fcntl,gc,hashlib,json,os,runpy,subprocess,sys,time
from collections import defaultdict,Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import numpy as np
import torch
import prefix_staged as p
import redesign_lib as lib
from common import Predictor,AttnProbe
from encoder import Encoder as LegacyEncoder
class Encoder(LegacyEncoder):
 @torch.no_grad()
 def initial(self,codes):
  # Preserve the validated single-image kernel shape, including the final partial batch.
  return np.concatenate([super(Encoder,self).initial(codes[i:i+1]) for i in range(len(codes))])
from support_utility import grouped_partition
from prefix_policy_predictor import PrefixPredictor,choose_action

R=Path(__file__).resolve().parents[1]; ASSETS=R/"assets_fixed"; W=R/'within_training'; P=R.parents[1]
OLD=P/'results/prefix_staged_20261008'; DATA=Path('/mnt/meg/jjenchen/phyre-unc')
PCA=DATA/'latents/levjepa_overlap_g32_pca128/pca.pt'
PHYRE='/path/to/research-assets/miniconda3/envs/phyre/bin/python'
f=p.f
read=f.read;write=f.write_json;sha=f.digest
BASES={'cross':DATA/'runs/cross_f1_base_s16r16_pca128/predictor.pt','within':DATA/'runs/base_s16r16_pca128/predictor.pt'}
PROBES={'cross':P/'results/probes_pca128_cross/ball_cross_template_f1_full.pt','within':P/'results/probes_pca128/ball_within_template_f0_full.pt'}
HEAD=P/'results/probe_adaptation_cross_20261001/heads/predicted_mix_s0/probe.pt'

def configure_within():
 lib.BASE=BASES['within'];lib.PROBE=PROBES['within']
 lib.PACKS=['within_f0_train_clusters_pca128','within_f0_train_s16r16_pca128']
 f.load_heads=lambda m,prior,original:({'original':original},{'original':m['probe_sha256']})

def load_base(panel):
 ck=torch.load(BASES[panel],map_location='cpu',weights_only=False);a=ck['args']
 base=Predictor(ck['Din'],ck['T'],ck['S'],a['d'],a['layers'],a['heads']).cuda()
 base.load_state_dict(ck['state_dict']);base.eval().requires_grad_(False)
 qc=torch.load(PROBES[panel],map_location='cpu',weights_only=False)
 hc=torch.load(HEAD,map_location='cpu',weights_only=False) if panel=='cross' else qc
 head=AttnProbe(hc['D'],hc['S']).cuda();head.load_state_dict(hc['state_dict']);head.eval().requires_grad_(False)
 return base,head,qc['rule'],ck

def prepare():
 assert not (R/'PLAN.json').exists() and not (W/'manifest.json').exists()
 configure_within();sel=read(R/'selection.json');parent=read(OLD/'manifest.json')
 store=f.Store(include_features=False)
 source=f.Pack(str(DATA/'packs/within_f0_train_s16r16_pca128'))
 assert len(source.actions)==52349 and set(source.by)==set(sel['panels']['within']['train'])
 assert not set(source.by)&set(sel['panels']['within']['test'])
 clusterfile=P/'data/manifests/ball_within_template_f0_train_clusters.jsonl'
 clusters=defaultdict(list)
 for line in clusterfile.read_text().splitlines():
  r=json.loads(line)
  if 'cluster' in r:clusters[r['file']].append(int(r['cluster']))
 seen={};ids_by=defaultdict(list);aliases={}
 for i in range(len(store.files)):
  key=(str(store.tasks[i]),tuple(store.actions[i].tolist()))
  if key in seen:
   j=seen[key];assert store.labels[i]==store.labels[j]
   clusters[str(store.files[j])].extend(clusters.get(str(store.files[i]),[]));aliases[str(i)]=j
  else:seen[key]=i;ids_by[key[0]].append(i)
 train=set(t for tp in sorted({t[:5] for t in source.by}) for t in sorted([t for t in source.by if t[:5]==tp],key=lambda t:f.stable(('within0-stage2',t)))[:48])
 assert len(train)==1200
 rows=[];assignments={};counts={}
 for task in sorted(ids_by):
  roles,groups=grouped_partition(store,task,ids_by[task],clusters)
  assert roles['support'] and roles['query'],(task,{k:len(v) for k,v in roles.items()})
  assert not {groups[str(i)]['group'] for i in roles['support']}&{groups[str(i)]['group'] for i in roles['query']}
  assert not {tuple(store.actions[i]) for i in roles['support']}&{tuple(store.actions[i]) for i in roles['query']}
  rows.append({'task':task,'split':'train' if task in train else 'calibration','candidates':roles['support'],'query':roles['query'],
   'times':[sorted(np.random.default_rng([20261005,i,73]).choice(np.arange(1,32),8,replace=False).tolist()) for i in roles['query']]})
  assignments[task]=groups
 assert set(ids_by)==set(source.by)
 for split in ['train','calibration']:
  rr=[r for r in rows if r['split']==split]
  counts[split]={'scenes':len(rr),'templates':dict(Counter(r['task'][:5] for r in rr)),
   'supports':sum(len(r['candidates']) for r in rr),'positive_supports':sum(int(store.labels[r['candidates']].sum()) for r in rr),
   'queries':sum(len(r['query']) for r in rr),'positive_queries':sum(int(store.labels[r['query']].sum()) for r in rr)}
 write(W/'partition.json',{'rows':assignments,'aliases':aliases})
 m={k:copy.deepcopy(parent[k]) for k in ['architecture','stage1','stage2','training','seeds']}
 m.update(code=f.code_hashes(),base=str(BASES['within']),base_sha256=sha(BASES['within']),probe=str(PROBES['within']),probe_sha256=sha(PROBES['within']),
  heads={'original':sha(PROBES['within'])},primary_head='original',rows=rows,data_counts=counts,
  stage1_pack=str(DATA/'packs/within_f0_train_s16r16_pca128'),stage1_dev_pack=str(DATA/'packs/within_f0_dev_pca128'))
 n=len(source.actions);updates=(n//8)*8
 m['stage1'].update(n_train=n,updates=updates,presentations=updates*8)
 verify=[BASES['within'],PROBES['within'],clusterfile,W/'partition.json',R/'selection.json']
 verify += [DATA/'packs'/(name+'_meta.npz') for name in lib.PACKS+['within_f0_dev_pca128']]
 m['verified_files']={str(x):sha(x) for x in verify}
 m['store_files_sha256']=hashlib.sha256('\n'.join(store.files).encode()).hexdigest()
 write(W/'manifest.json',m);write(W/'DATA_AUDIT.json',{'passed':True,'manifest':sha(W/'manifest.json'),'counts':counts,'stage1_updates':updates,'near_filter_used':False,'all_train_scenes_retained':True})
 cross={}
 for s in ['q0','residual','zero']:
  for seed in range(3):
   d=read(OLD/'models'/f'{s}_s{seed}'/'complete.json');assert d['complete'] and sha(d['checkpoint'])==d['checkpoint_sha256']
   assert d['provenance']['manifest']==sha(OLD/'manifest.json')
   cross[f'{s}_s{seed}']={'checkpoint':d['checkpoint'],'sha256':d['checkpoint_sha256']}
 plan={'code':f.code_hashes(),'selection_sha256':sha(R/'selection.json'),'within_manifest_sha256':sha(W/'manifest.json'),
  'cross_parent_manifest_sha256':sha(OLD/'manifest.json'),'cross_checkpoints':cross,
  'base':{k:{'path':str(v),'sha256':sha(v)} for k,v in BASES.items()},
  'probe':{k:{'path':str(v),'sha256':sha(v)} for k,v in PROBES.items()},'cross_head':{'path':str(HEAD),'sha256':sha(HEAD)},
  'pca':{'path':str(PCA),'sha256':sha(PCA)},'seeds':[0,1,2],'budgets':[1,3,5,9,20],'primary':'S@9',
  'candidates':'First5000 cache actions, validity only; base+fixed panel scorer top1024, stable tie break; outcomes re-simulated',
  'scorer':{'cross':'predicted_mix_s0','within':'original within0 probe'},
  'observation':'Only selected failed trials encoded; truth observations shared between models, residuals always recomputed from current model/tokens',
  'gpu_limit':4,'stage1_budget':m['stage1'],'data_counts':counts}
 write(R/'PLAN.json',plan);print('PREPARED',json.dumps(counts),flush=True)

def checked():
 plan=read(R/'PLAN.json');assert plan['code']==f.code_hashes()
 assert plan['selection_sha256']==sha(R/'selection.json')
 assert plan['within_manifest_sha256']==sha(W/'manifest.json')
 assert plan['pca']['sha256']==sha(PCA)
 for r in list(plan['base'].values())+list(plan['probe'].values())+[plan['cross_head']]:assert sha(r['path'])==r['sha256']
 return plan

@torch.no_grad()
def encoding_check():
 encoder=Encoder(PCA);metrics=[]
 argv=sys.argv;sys.argv=['legacy','--encoder','levjepa','--manifests','unused','--pca',str(PCA)]
 legacy=runpy.run_path(str(R/'code/legacy_encode_latents.py'),run_name='legacy');sys.argv=argv
 for name in ['within_f0_train_s16r16_pca128','cross_f1_train_s16r16_pca128']:
  with np.load(DATA/'packs'/(name+'_meta.npz')) as z:file=str(z['files'][0])
  rawpath=P/'data'/file
  if not rawpath.exists():rawpath=DATA/'data'/file
  with np.load(rawpath) as z:frames=z['frames']
  full=encoder.rollout(frames);initial=encoder.initial(frames[:1])[0];truth=np.load(DATA/'packs'/(name+'_latents.npy'),mmap_mode='r')[0]
  err=float(np.abs(full.astype(float)-truth).max());err0=float(np.abs(initial.astype(float)-full[0]).max())
  steps=legacy['encode_rollout'](frames)
  reproduced=encoder.project(steps[list(range(0,128,4))].reshape(32,196,-1)).cpu().numpy()
  np.testing.assert_array_equal(full,reproduced)
  mean_error=float(np.abs(full.astype(float)-truth).mean())
  mean_initial=float(np.abs(initial.astype(float)-full[0]).mean())
  # Exact legacy implementation match; historical fp16 rounding differs sparsely.
  assert err<.01 and mean_error<1e-6 and err0<.01 and mean_initial<1e-6,(name,err,err0,mean_error,mean_initial)
  metrics.append({'pack':name,'exact_new_vs_legacy':True,'max_abs_full_vs_cached':err,'max_abs_initial_vs_full':err0,'mean_abs_full_vs_cached':mean_error,'mean_abs_initial_vs_full':mean_initial})
 rawpath=next((R/'assets/cross/observations').glob('.raw*.npz'))
 with np.load(rawpath) as z:frames=z['frames']
 full=encoder.rollout(frames);batch=encoder.initial(np.repeat(frames[:1],32,axis=0))
 diff=np.abs(batch.astype(float)-full[0]);assert diff.max()<.01 and diff.mean()<1e-6
 metrics.append({'real_case_32_image_batch_max':float(diff.max()),'mean':float(diff.mean())})
 write(R/'ENCODING_CHECK_FIXED.json',{'passed':True,'code':f.code_hashes(),'metrics':metrics})
 return encoder

@torch.no_grad()
def shortlist(shard):
 plan=checked();assert read(R/'ENCODING_CHECK_FIXED.json')['passed']
 encoder=Encoder(PCA);sel=read(R/'selection.json')
 for panel,v in sel['panels'].items():
  base,head,rule,_=load_base(panel);score=p.score_function(head,rule)
  for task in v['selected'][shard::4]:
   src=R/'assets'/panel/'candidates'/(task.replace(':','_')+'.npz');out=ASSETS/panel/'shortlist'/src.name
   if out.exists():continue
   with np.load(src) as zz:d={k:zz[k] for k in zz.files}
   latent=[];scores=[]
   for start in range(0,len(d['action']),32):
    z=encoder.initial(d['frames'][start:start+32]);latent.append(z)
    z0=torch.from_numpy(z).cuda().float();a=torch.from_numpy(d['action'][start:start+32]).cuda()
    feats=[v[None] for v in d['features'][start:start+32]]
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=base(z0,a)
    scores.append(score(pred,feats).cpu().numpy())
   scores=np.concatenate(scores);latent=np.concatenate(latent);assert np.isfinite(scores).all()
   ids=np.argsort(-scores,kind='stable')[:1024];assert len(ids)==1024
   out.parent.mkdir(parents=True,exist_ok=True)
   f.save_npz(out,{'z0':latent[ids],'features':d['features'][ids],'action':d['action'][ids],
    'action_idx':d['action_idx'][ids],'selection_score':scores[ids],'task':np.array(task),'plan_sha256':np.array(sha(R/'PLAN.json')),'runtime_sha256':np.array(sha(__file__))})
   print('SHORTLIST',panel,task,'valid',len(scores),flush=True)
  del base,head;gc.collect();torch.cuda.empty_cache()
 write(R/f'SHORTLIST_FIXED_{shard}_COMPLETE.json',{'complete':True,'plan_sha256':sha(R/'PLAN.json')})

class Observations:
 def __init__(self,encoder=None):
  self.encoder=encoder;self.worker=None
 def get(self,panel,task,action_idx,action,z0,expected_status):
  folder=ASSETS/panel/'observations';folder.mkdir(parents=True,exist_ok=True)
  path=folder/f'{task.replace(":","_")}_{action_idx:06d}.npz'
  with open(str(path)+'.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if not path.exists():
    if self.worker is None:
     self.worker=subprocess.Popen([PHYRE,'-u',str(R/'code/data_cpu.py'),'serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    raw=folder/f'.raw_{os.getpid()}_{task.replace(":","_")}_{action_idx}.npz'
    self.worker.stdin.write(json.dumps({'task':task,'action':action.tolist(),'out':str(raw)})+'\n');self.worker.stdin.flush()
    while True:
     line=self.worker.stdout.readline()
     if not line:raise RuntimeError('PHYRE observation worker exited')
     try:response=json.loads(line)
     except json.JSONDecodeError:continue
     assert response['ok'] and response['status']==int(expected_status);break
    if self.encoder is None:self.encoder=Encoder(PCA)
    with np.load(raw) as z:
     latent=self.encoder.rollout(z['frames']);features=z['features'][::4];status=int(z['status'])
    diff=np.abs(latent[0].astype(float)-z0);error=float(diff.max())
    assert error<.01 and diff.mean()<1e-6,(error,float(diff.mean()))
    # Preserve true encoded t0; current residual is observation minus current prediction.
    f.save_npz(path,{'latents':latent,'features':features,'action':action,'status':np.array(status),
     'task':np.array(task),'pca_sha256':np.array(sha(PCA)),'initial_encoding_maxdiff':np.array(error)})
    raw.unlink()
   with np.load(path) as z:
    assert str(z['task'])==task and int(z['status'])==int(expected_status) and str(z['pca_sha256'])==sha(PCA)
    np.testing.assert_array_equal(z['action'],action);truth=z['latents'].copy()
   return torch.from_numpy(truth[None]).cuda().float()
 def close(self):
  if self.worker:
   self.worker.stdin.close();self.worker.wait(timeout=60);self.worker=None

def model_for(panel,stage,seed,smoke=False):
 plan=checked()
 if panel=='within':
  configure_within();folder=W/('smoke' if smoke else 'models')
  d=read(folder/f'{stage}_s{seed}'/'complete.json');assert sha(d['checkpoint'])==d['checkpoint_sha256']
  parent=Path(read(folder/f'q0_s{seed}'/'complete.json')['checkpoint']) if stage!='q0' else None
  ref,probe,rule,model,head,_=p.model_for(W,stage,seed,Path(d['checkpoint']),parent)
  return ref,model,head,rule,d['checkpoint_sha256']
 ref,head,rule,ck=load_base(panel);d=plan['cross_checkpoints'][f'{stage}_s{seed}']
 assert sha(d['checkpoint'])==d['sha256']
 mc=torch.load(d['checkpoint'],map_location='cpu',weights_only=False)
 assert mc['seed']==seed and mc['stage']==stage and mc['provenance']['manifest']==plan['cross_parent_manifest_sha256']
 model=PrefixPredictor(copy.deepcopy(ref),ck['Din'],stage='initial' if stage=='q0' else 'updater',zero_residual=stage=='zero',**read(OLD/'manifest.json')['architecture']).cuda()
 model.load_state_dict(mc['state_dict']);return ref,model,head,rule,d['sha256']

def input_case(panel,task):
 path=ASSETS/panel/'shortlist'/(task.replace(':','_')+'.npz')
 with np.load(path) as z:d={k:z[k] for k in z.files}
 assert str(d['task'])==task and str(d['plan_sha256'])==sha(R/'PLAN.json')
 assert str(d['runtime_sha256'])==sha(__file__)
 status=np.load(path.with_name(path.stem+'_labels.npy'));assert status.shape==(1024,) and (status!=0).all()
 return d,status,path

@torch.no_grad()
def evaluate(panel,stage,seed,obs,smoke=False,tasks=None,budget=20):
 ref,model,head,rule,checkpoint_sha=model_for(panel,stage,seed,smoke);model.eval().requires_grad_(False)
 basefolder=R/('smoke_evaluation_fixed' if smoke else 'evaluation_fixed')/panel
 folder=basefolder/f'{stage}_s{seed}';score=p.score_function(head,rule)
 prov={'plan':sha(R/'PLAN.json'),'checkpoint':checkpoint_sha,'panel':panel,'stage':stage,'seed':seed,'smoke':smoke,'runtime_sha256':sha(__file__)}
 tasks=tasks or read(R/'selection.json')['panels'][panel]['selected'];results=[]
 for task in tasks:
  out=folder/task.replace(':','_');d,status,path=input_case(panel,task)
  if (out/'result.json').exists():
   rr=read(out/'result.json');assert rr['provenance']==prov;p.audit_case(out);results.append(rr);continue
  out.mkdir(parents=True,exist_ok=True);labels=status>0
  z0=torch.from_numpy(d['z0']).cuda().float();actions=torch.from_numpy(d['action']).cuda();features=[v[None] for v in d['features']]
  if stage=='q0':
   original=p.static_scores(ref,z0,actions,features,score);qs=p.static_scores(model,z0,actions,features,score)
   raw={'indices':d['action_idx'],'labels':labels,'original':original,'q0':qs}
   rr={'task':task,'stage':stage,'provenance':prov,'first_success':p.first_rank(qs,labels),'original_first_success':p.first_rank(original,labels)}
  else:
   state=model.initial_state().clone();tried=torch.zeros(len(actions),device='cuda',dtype=torch.bool);snapshots=[];selected=[];success=None
   for attempt in range(1,budget+1):
    with torch.autocast('cuda',dtype=torch.bfloat16):chosen,scores,pred=choose_action(model,state,z0,actions,features,score,tried)
    assert chosen is not None and not tried[chosen];tried[chosen]=True
    selected.append(chosen);snapshots.append(scores.cpu().numpy())
    if labels[chosen]:success=attempt;break
    if attempt==budget:break
    # Outcome is inspected only after argmax. Only failed selected actions can update tokens.
    if stage=='zero':truth=None
    else:truth=obs.get(panel,task,int(d['action_idx'][chosen]),d['action'][chosen],d['z0'][chosen],status[chosen])
    with torch.autocast('cuda',dtype=torch.bfloat16):
     residual=torch.zeros_like(pred) if truth is None else truth-pred
     state=model.update_state(state,residual,actions[chosen:chosen+1])
   with np.load(basefolder/f'q0_s{seed}'/task.replace(':','_')/'scores.npz') as q:
    np.testing.assert_array_equal(q['indices'],d['action_idx']);np.testing.assert_array_equal(q['labels'],labels)
    np.testing.assert_array_equal(q['q0'],snapshots[0])
   raw={'indices':d['action_idx'],'labels':labels,'scores':np.stack(snapshots),'selected':np.array(selected)}
   rr={'task':task,'stage':stage,'provenance':prov,'first_success':success,'attempts':len(selected),'updates':len(selected)-1,'budget':budget}
  rr.update(shortlist_sha256=sha(path),labels_sha256=sha(path.with_name(path.stem+'_labels.npy')))
  f.save_npz(out/'scores.npz',raw);write(out/'result.json',rr);p.audit_case(out);results.append(rr)
  write(folder/'progress.json',{'scenes':len(results),'total':len(tasks),'last_task':task})
  print('EVAL',panel,stage,seed,len(results),task,rr['first_success'],flush=True)
 write(folder/'complete.json',{'complete':True,'n':len(results),'provenance':prov,
  'counts':{f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in results) for k in [1,3,5,9,20]}})
 del model,ref,head;gc.collect();torch.cuda.empty_cache()

def smoke_train():
 checked();configure_within()
 q=p.train(W,'q0',0,True);parent=Path(q['checkpoint'])
 branches=[p.train(W,s,0,True,parent) for s in ['residual','zero']]
 assert branches[0]['frozen_sha256']==branches[1]['frozen_sha256']
 assert branches[0]['initial_model_sha256']==branches[1]['initial_model_sha256']
 stress=p.stress_replay(W,parent,Path(branches[0]['checkpoint']))
 write(W/'SMOKE.json',{'passed':True,'provenance':f.provenance(W),'stages':[q]+branches,'max_depth_stress':stress})
 print('TRAIN_SMOKE_PASS',flush=True)

def smoke_eval():
 checked();obs=Observations()
 for panel in ['cross','within']:
  # Engineering-only choose a task with enough failed actions; no performance selection.
  tasks=read(R/'selection.json')['panels'][panel]['selected'];task=tasks[0]
  for stage in ['q0','residual','zero']:evaluate(panel,stage,0,obs,smoke=True,tasks=[task],budget=3)
  # Explicitly exercise lazy encoding even if the policy solves on trial one.
  d,status,_=input_case(panel,task);j=int(np.flatnonzero(status<0)[0])
  truth=obs.get(panel,task,int(d['action_idx'][j]),d['action'][j],d['z0'][j],status[j]);assert truth.shape==(1,32,196,128)
 obs.close();write(R/'EVAL_SMOKE_FIXED.json',{'passed':True,'plan':sha(R/'PLAN.json')})

def lane(panel,seed):
 checked();assert read(R/'EVAL_SMOKE_FIXED.json')['passed']
 if panel=='within':
  configure_within()
  for stage in ['q0','residual','zero']:p.train(W,stage,seed)
 obs=Observations()
 for s in (range(3) if panel=='cross' else [seed]):
  for stage in ['q0','residual','zero']:evaluate(panel,stage,s,obs)
 obs.close()

def summary():
 plan=checked();out={}
 for panel in ['cross','within']:
  out[panel]={};tasks=read(R/'selection.json')['panels'][panel]['selected']
  for seed in range(3):
   for stage in ['q0','residual','zero']:
    folder=R/'evaluation_fixed'/panel/f'{stage}_s{seed}';done=read(folder/'complete.json');assert done['n']==100
    rows=[]
    for task in tasks:
     sub=folder/task.replace(':','_');r=p.audit_case(sub);d,labels,path=input_case(panel,task)
     assert r['shortlist_sha256']==sha(path) and r['labels_sha256']==sha(path.with_name(path.stem+'_labels.npy'))
     with np.load(sub/'scores.npz') as z:
      np.testing.assert_array_equal(z['indices'],d['action_idx']);np.testing.assert_array_equal(z['labels'],labels>0)
      with np.load(R/'evaluation_fixed'/panel/f'q0_s{seed}'/task.replace(':','_')/'scores.npz') as q:
       if stage!='q0':np.testing.assert_array_equal(z['scores'][0],q['q0'])
      if stage=='q0' and seed:
       with np.load(R/'evaluation_fixed'/panel/'q0_s0'/task.replace(':','_')/'scores.npz') as q:np.testing.assert_array_equal(z['original'],q['original'])
     rows.append(r)
    counts={f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in rows) for k in plan['budgets']};assert counts==done['counts']
    out[panel][f'{stage}_s{seed}']={'counts':counts,'first_success':{r['task']:r['first_success'] for r in rows}}
    if stage=='q0':out[panel][f'original_s{seed}']={'counts':{f's{k}':sum(r['original_first_success'] is not None and r['original_first_success']<=k for r in rows) for k in plan['budgets']}}
 for seed in range(3):
  q=read(W/'models'/f'q0_s{seed}'/'complete.json');assert q['slots']==q['optimizer_steps']==plan['stage1_budget']['updates']
  a,b=[read(W/'models'/f'{stage}_s{seed}'/'complete.json') for stage in ['residual','zero']]
  assert a['initial_model_sha256']==b['initial_model_sha256'] and a['frozen_sha256']==b['frozen_sha256']
  for r in [a,b]:assert r['parent_checkpoint_sha256']==q['checkpoint_sha256'] and r['slots']==10000
 write(R/'SUMMARY_FIXED.json',{'complete':True,'raw_audit_passed':True,'plan':sha(R/'PLAN.json'),'splits':out})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--panel',default='within');a=ap.parse_args()
 torch.set_num_threads(4)
 if a.command=='prepare':prepare()
 elif a.command=='encoding_check':encoding_check()
 elif a.command=='shortlist':shortlist(a.shard)
 elif a.command=='smoke_train':smoke_train()
 elif a.command=='smoke_eval':smoke_eval()
 elif a.command=='lane':lane(a.panel,a.seed)
 elif a.command=='summary':summary()
 else:raise ValueError(a.command)
