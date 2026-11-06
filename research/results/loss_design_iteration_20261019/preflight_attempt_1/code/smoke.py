from common import *
import argparse

def run(panel):
 import torch
 src=source();checked();team=make_team(panel);load(team,panel)
 try:
  verified=False
  for i in range(32):
   ep=team.collect(i,True,cap_override=2)
   if len(ep['supports'])<2:continue
   tests=[]
   for old in [True,False]:
    team.model.zero_grad(set_to_none=True);team.model.checkpoint_blocks=False
    with src.controlled_backward('cpu',True):
     loss,_,_=src.Team.loss(team,ep,True,True) if old else team.loss(ep,True,True);loss.backward()
    tests.append((float(loss),src.old.v.fingerprint((n,p.grad) for n,p in team.model.named_parameters() if p.grad is not None)))
   assert tests[0]==tests[1];verified=True;break
  assert verified;write(R/f'LOSS_REPLAY_{panel}.json',{'passed':True,'nonidentity_two_update_l1_loss_gradient_exact':True,'plan_sha256':sha(R/'PLAN.json')})
 finally:close(team)
 from evaluate import run as evaluate
 evaluate(panel,'l1',0,'raw',0,1,True)

def summary():
 from evaluate import collect
 records={}
 for panel in ['cross','within']:
  assert read(R/f'LOSS_REPLAY_{panel}.json')['passed'];records[panel]=collect(panel,'l1',0,'raw',True)
 write(R/'SMOKE.json',{'passed':True,'plan_sha256':sha(R/'PLAN.json'),'loss_gradient_exact':True,'full_raw_policy_replay_exact':True,'records':records});print('SMOKE_PASSED')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--panel');a.add_argument('--summary',action='store_true');a=a.parse_args();summary() if a.summary else run(a.panel)
