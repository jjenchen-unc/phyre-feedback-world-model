from loss_runtime import *
import argparse,numpy as np

def run(panel,kind,seed):
 import torch
 src=source();plan=checked();assert read(R/'SMOKE.json')['passed'];cal=read(R/f'CALIBRATION_{panel}.json');assert cal['complete']
 team=make_team(panel,seed);configure(team,panel,kind);model=team.model;initial=src.old.ident(model);reader=src.old.reader_hash(model);probe=src.old.ident(team.probe);head=src.old.ident(team.head)
 folder=R/'models'/f'seed{seed}'/panel/kind;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():assert read(folder/'complete.json')['plan_sha256']==sha(R/'PLAN.json');close(team);return
 assert not (folder/'episodes.jsonl').exists(),'No partial restart'
 ref=read(B/'models'/panel/'uniform/preflight.json');assert reader==ref['reader_sha256']
 if seed==0:assert initial==ref['initial_sha256']
 # Independent episode seed by fixed index namespace; pairs share it exactly.
 offset=100000*seed;opt,sched=src.optimizer(model,128);began=time.time()
 try:
  # Genuine nonidentity/current-residual, two-update loss/gradient replay.
  fresh={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};load(team,panel);verified=False
  for i in range(32):
   ep=team.collect(offset+i,True,cap_override=2)
   if len(ep['supports'])<2:continue
   tests=[]
   for whole in [False,True]:
    model.zero_grad(set_to_none=True);model.checkpoint_blocks=not whole
    with src.controlled_backward('cpu',True):loss,_,_=team.loss(ep,whole,True);loss.backward()
    assert model.residual_in.weight.grad is not None and model.residual_in.weight.grad.abs().sum()>0
    tests.append((float(loss),src.old.v.fingerprint((n,p.grad) for n,p in model.named_parameters() if p.grad is not None)))
    del loss
   assert tests[0]==tests[1];verified=True;break
  assert verified;model.load_state_dict(fresh);del fresh;model.zero_grad(set_to_none=True);model.checkpoint_blocks=False;assert src.old.ident(model)==initial
  write(folder/'preflight.json',{'passed':True,'two_update_loss_gradient_exact':True,'initial_sha256':initial,'reader_sha256':reader,'scale':team.loss_scale,'beta':team.smooth_beta,'plan_sha256':sha(R/'PLAN.json'),'seconds':time.time()-began})
  reference=[json.loads(l) for l in (B/'models'/panel/'uniform/episodes.jsonl').read_text().splitlines()];updates=0;began=time.time()
  with (folder/'episodes.jsonl').open('w',buffering=1) as f:
   for slot in range(128):
    t=time.time();opt.zero_grad(set_to_none=True);events=[];active=False
    for e in range(2):
     ep=team.collect(offset+2*slot+e);loss,l1,geo=team.loss(ep,True,slot<3);assert torch.isfinite(loss)
     if loss.requires_grad:(loss/2).backward();active=True
     event=team.event(ep,loss,l1,geo);event['query_positive_count']=ep['query_positive_count'];events.append(event);del ep,loss
    if seed==0:
     assert active==reference[slot]['optimizer_step']
     for e,refevent in zip(events,reference[slot]['events']):assert (e['task'],e['cap'],e['selected'][0],bool(e['supports']))==(refevent['task'],refevent['cap'],refevent['selected'][0],bool(refevent['supports']))
    norm=None
    if active:
     norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(norm) and norm>0;opt.step();updates+=1
    sched.step();done=slot+1;f.write(json.dumps({'slot':slot,'optimizer_step':active,'events':events,'gradient_norm_before_clip':float(norm) if norm is not None else None,'seconds':time.time()-t})+'\n')
    if done%4==0:
     assert src.old.reader_hash(model)==reader and src.old.ident(team.probe)==probe and src.old.ident(team.head)==head
     write(folder/'progress.json',{'slots':done,'optimizer_steps':updates,'elapsed_s':time.time()-began});print('TRAIN',panel,kind,seed,done,round(time.time()-began,1),flush=True)
    if done in [64,128]:
     path=folder/f'model_{done}.pt';src.old.b.atomic_save(path,{'state_dict':model.state_dict(),'slots':done,'plan_sha256':sha(R/'PLAN.json')});receipt={'complete':True,'panel':panel,'loss_kind':kind,'seed':seed,'slots':done,'checkpoint':str(path),'checkpoint_sha256':sha(path),'initial_sha256':initial,'reader_sha256':reader,'optimizer_steps':updates,'plan_sha256':sha(R/'PLAN.json'),'calibration_sha256':sha(R/f'CALIBRATION_{panel}.json'),'scale':team.loss_scale,'beta':team.smooth_beta,'elapsed_s':time.time()-began};write(folder/f'checkpoint_{done}.json',receipt)
     if done==128:write(folder/'complete.json',receipt)
 finally:close(team)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--panel',required=True);a.add_argument('--kind',choices=['l1','mse','smooth'],required=True);a.add_argument('--seed',type=int,default=0);a=a.parse_args();run(a.panel,a.kind,a.seed)
