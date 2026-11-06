"""Adaptive target-specific geometry supervision, matched probe/times/data controls."""
import sys,argparse,importlib.util,time,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import study as st
spec=importlib.util.spec_from_file_location('contact_train_reference',ROOT/'code/train.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)
import torch
from scorer_torch import distances
from common import surface_dist
RR=ROOT/'contact_training';read,write,sha=st.read,st.write,st.sha
ARMS=['contact01_residual','contact01_zero','contact1_residual','contact1_zero']


def register():
 p=st.checked();diag=read(ROOT/'CONTACT_TARGET_DIAGNOSTIC.json');assert diag['complete']
 plan=dict(parent=sha(ROOT/'PLAN.json'),source_sha=sha(Path(__file__)),trigger_sha=sha(ROOT/'CONTACT_TARGET_DIAGNOSTIC.json'),arms=ARMS,weights={'contact01':.1,'contact1':1.},seed=7919,steps=128,
   budget_amendment='Original12 updaterfits completed. Add exactly4fresh contact-target fits, total16; no extra pretraining/new simulations/encoders. User authorized iterative small experiments; neworacle direction diagnostic motivates this separate amendment.',
   trigger='Binary success objectives and scalaradaptation didnotgive stablecrossgains; observed-geometryoracle and partialdistancecorrection suggest task-relative physical target. Oracle is privileged, not evidence of updaterlearnability.',
   loss='Original L1future+.1objectgeometry + lambda*mean(abs(predicted_goal_surface_distance-true_goal_surface_distance))/256.',
   isolation='Contact distance uses SAME ORIGINAL geometry probe and SAME8sampledtimepoints as originalgeometryloss, perquery probe shapes preserved. All8uniformqueries, no success-label/near filtering, no additionaldata. Rankingprobe/scorer unchanged. Target is derived from alreadyavailablegeometry supervision.',
   schedule='Freshseed7919 updater, identical128scene/querydraws and optimizer as round1; centeredreal/zero paired. FixedQ0reader/initialtokens/probes. Firstsupport fixedQ0failure; currentfirstpredictionexactcache.',
   evaluation='Both weights/conditions on discovery16,confirmation16 andexistingcross100. No choosingbesttestcheckpoint; onlyfinal128steps. Compare ownoriginalprediction-loss real/zero; geometryandranking separately. Singlefeedback nextk, notmultiturnS9.',
   limits=['One updaterseed for this newtarget','Two fixed physical-loss weights, no tuning on test','Usesoriginalgeometryprobe, notadapted rankingprobe for gradient; same asoriginalgeometryloss','Favorable oraclecorrection doesnotprove supportcontainsneededqueryspecific information'])
 assert not(RR/'PLAN.json').exists();write(RR/'PLAN.json',plan)


def check():
 p=read(RR/'PLAN.json');assert p['source_sha']==sha(Path(__file__)) and p['parent']==sha(ROOT/'PLAN.json');st.checked();return p


def contact_loss(team,pred,row):
 ts=row['times'];values=[]
 for j,f in enumerate(row['features']):
  d=distances(team.probe,pred[j:j+1,ts], [f[:1]])[0]
  target=torch.from_numpy(surface_dist(f)[ts].copy()).cuda().float()
  values.append((d-target).abs().mean()/256)
 return torch.stack(values).mean()


def smoke():
 p=check();team=st.team_new();cal=read(ROOT/'DATA_AND_CALIBRATION.json');records=[];initial=st.ar.lr.source().old.ident(team.model)
 try:
  rows=[st.data(t) for t in st.checked()['fit']];rows=[r for r in rows if r['failed']][:8]
  for row in rows:
   state=team.model.initial_state().detach().clone().requires_grad_(True);losses,pred,_=tr.terms(team,row,state,cal);contact=contact_loss(team,pred,row)
   g=torch.autograd.grad(contact,state,retain_graph=True)[0];base=torch.autograd.grad(losses['prediction'],state)[0];assert torch.isfinite(g).all() and g.norm()>0
   records.append(dict(task=row['task'],contact_loss=float(contact.detach()),prediction_loss=float(losses['prediction'].detach()),contact_grad=float(g.norm()),prediction_grad=float(base.norm()),ratio=float(g.norm()/base.norm())))
   del state,losses,pred,contact,g,base
  row=rows[0];state=tr.state_for(team,row);losses,pred,_=tr.terms(team,row,state,cal);loss=losses['prediction']+.1*contact_loss(team,pred,row);loss.backward()
  assert any(p.grad is not None and p.grad.norm()>0 for p in team.model.parameters() if p.requires_grad)
  assert all(p.grad is None for p in team.model.parameters() if not p.requires_grad)
  assert all(p.grad is None for h in [team.probe,team.head] for p in h.parameters())
  assert initial==st.ar.lr.source().old.ident(team.model)
  write(RR/'SMOKE.json',dict(passed=True,plan=sha(RR/'PLAN.json'),records=records,initial_unchanged=True,same_probe_and_timepoints=True));print('CONTACT_SMOKE',records,flush=True)
 finally:st.ar.lr.close(team)


def train(index):
 p=check();assert read(RR/'SMOKE.json')['passed'];arm=ARMS[index];kind,cond=arm.split('_');weight=p['weights'][kind];team=st.team_new(p['seed'],cond);model=team.model;folder=RR/'models'/arm
 assert not(folder/'complete.json').exists() and not(folder/'progress.json').exists()
 cal=read(ROOT/'DATA_AND_CALIBRATION.json');rows=[st.data(t) for t in st.checked()['fit']];rows=[r for r in rows if r['failed']]
 rng=np.random.default_rng([20261021,p['seed']]);order=[]
 while len(order)<p['steps']:order+=rng.permutation(len(rows)).tolist()
 initial=st.ar.lr.source().old.ident(model);reader=st.ar.lr.source().old.reader_hash(model);params=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=.05);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['steps']);events=[];start=time.time()
 try:
  for step,idx in enumerate(order[:p['steps']]):
   row=rows[idx];opt.zero_grad(set_to_none=True);state=tr.state_for(team,row);losses,pred,_=tr.terms(team,row,state,cal);contact=contact_loss(team,pred,row);loss=losses['prediction']+weight*contact
   assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(gn);opt.step();sched.step()
   events.append(dict(step=step+1,task=row['task'],positive=sum(row['labels']),loss=float(loss.detach()),contact=float(contact.detach()),prediction=float(losses['prediction'].detach()),geometry=float(losses['geometry'].detach()),grad_norm=float(gn)))
   del state,losses,pred,contact,loss
   if(step+1)%8==0:write(folder/'progress.json',events[-1]);print('CONTACT_TRAIN',arm,events[-1],flush=True)
  assert reader==st.ar.lr.source().old.reader_hash(model)
  names={n for n,p in model.named_parameters() if p.requires_grad};folder.mkdir(parents=True,exist_ok=True);path=folder/'updater.pt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k in names},path)
  write(folder/'events.json',events);write(folder/'complete.json',dict(complete=True,plan=sha(RR/'PLAN.json'),initial_sha=initial,reader_sha=reader,checkpoint_sha=sha(path),steps=p['steps'],seconds=time.time()-start,query_schedule_sha=hashlib.sha256(json.dumps([(e['task'],e['positive']) for e in events]).encode()).hexdigest()))
 finally:st.ar.lr.close(team)


def evaluate(stage,shard,nshards):
 from fast_inference import PreparedScorer
 p=check();team=st.team_new();scorer=PreparedScorer(team.head,team.rule);cal=read(ROOT/'DATA_AND_CALIBRATION.json');reader=st.ar.lr.source().old.reader_hash(team.model)
 tasks=read(ROOT/'cross_transfer/EXTENSION_PLAN.json')['all_tasks'] if stage=='cross100' else st.checked()[stage]
 base=ROOT/'cross_transfer/data' if stage=='cross100' else ROOT/'data'
 for arm in ARMS:assert read(RR/'models'/arm/'complete.json')['complete']
 try:
  for task in tasks[shard::nshards]:
   out=RR/'evaluation'/stage/task.replace(':','_');d=base/task.replace(':','_');r=read(d/'complete.json')
   if(out/'result.json').exists():assert read(out/'result.json')['plan']==sha(RR/'PLAN.json');continue
   if not r['failed']:
    write(out/'result.json',dict(complete=True,plan=sha(RR/'PLAN.json'),task=task,first_failed=False));continue
   row=dict(support_action=r['support_action'],support_prediction=np.load(d/'support_prediction.npy'),support_observed=np.load(d/'support_observed.npy')) if stage=='cross100' else st.data(task)
   team.load(task);arrays={};lossrec={}
   with torch.no_grad():
    for arm in ARMS:
     folder=RR/'models'/arm;ck=read(folder/'complete.json');assert ck['checkpoint_sha']==sha(folder/'updater.pt')
     missing,extra=team.model.load_state_dict(torch.load(folder/'updater.pt',map_location='cpu',weights_only=True),strict=False);assert not extra and all(n.startswith('base.') or n=='initial_tokens' for n in missing)
     team.model.eval().requires_grad_(False);team.model.checkpoint_blocks=False;team.input_mode='centered' if arm.endswith('residual') else 'zero';state=tr.state_for(team,row);chunks=[]
     for start in range(0,1024,32):
      z,a=team.chunk(start)
      with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a,state.expand(32,-1,-1))
      chunks.append(scorer(pred,[f[None] for f in team.pool['features'][start:start+32]]).cpu().numpy())
     arrays[arm]=np.concatenate(chunks)
     if stage!='cross100':
      ls,pred,_=tr.terms(team,row,state,cal,grad=False);lossrec[arm]={k:float(v) for k,v in ls.items()};lossrec[arm]['contact']=float(contact_loss(team,pred,row))
   assert reader==st.ar.lr.source().old.reader_hash(team.model)
   st.ar.lr.source().data.save_npz(out/'scores.npz',**arrays,labels=team.pool['status']>0,indices=team.pool['action_idx'],first=np.array(r['first']))
   write(out/'result.json',dict(complete=True,plan=sha(RR/'PLAN.json'),task=task,first_failed=True,reader_unchanged=True,losses=lossrec));print('CONTACT_EVAL',stage,task,flush=True)
 finally:st.ar.lr.close(team)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','smoke','train','discovery','confirmation','cross100']);p.add_argument('--index',type=int,default=0);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=16);a=p.parse_args()
 if a.mode=='register':register()
 elif a.mode=='smoke':smoke()
 elif a.mode=='train':train(a.index)
 else:evaluate(a.mode,a.shard,a.nshards)
