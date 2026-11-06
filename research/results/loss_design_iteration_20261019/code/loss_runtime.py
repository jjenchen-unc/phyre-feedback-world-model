"""Isolated loss ablation; immutable reader, current-policy residual, uniform queries."""
import hashlib, importlib.util, json, os, sys, time, types
from pathlib import Path
P=Path(__file__).resolve().parents[3]; R=P/'results/loss_design_iteration_20261019'
C=P/'results/centered_residual_training_20261019'; B=P/'results/query_balance_training_20261019'; S=P/'results/q0_first_feedback_20261018'
spec=importlib.util.spec_from_file_location('center_reference',C/'code/train.py');ct=importlib.util.module_from_spec(spec);spec.loader.exec_module(ct);ct.R=R
read=ct.read;sha=ct.sha

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)

def checked():
 p=read(R/'PLAN.json')
 for path,h in p['dependencies'].items():assert sha(path)==h,path
 return p

def source():
 import torch,selector_feedback_run as src
 torch.set_num_threads(2);assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
 return src

def make_team(panel,seed=0,mode='raw'):
 import torch,copy
 src=source();team=ct.make_team(panel,'centered',src.data.checked(),mode)
 team.truth.parents.insert(0,C/'observations'/panel)
 if seed:
  torch.manual_seed(7919+seed)
  fresh=src.old.b.PrefixPredictor(copy.deepcopy(team.model.base),team.model.base.inp[1].in_features,stage='updater',checkpoint_blocks=False).cuda()
  with torch.no_grad():fresh.initial_tokens.copy_(team.model.initial_tokens)
  assert src.old.reader_hash(fresh)==src.old.reader_hash(team.model)
  team.center_hook.remove();team.model=fresh.eval();team.center_hook=fresh.residual_in.register_forward_pre_hook(lambda module,args:(ct.transform(args[0],team.input_mode),))
 team.loss_kind='l1';team.loss_scale=1.;team.smooth_beta=1.
 team.loss=types.MethodType(loss,team)
 return team

def latent_terms(pred,target,beta):
 import torch.nn.functional as F
 e=pred[:,1:].float()-target[:,1:].float()
 return {'l1':e.abs().mean(),'mse':e.square().mean(),'smooth':F.smooth_l1_loss(pred[:,1:].float(),target[:,1:].float(),beta=beta)}

def loss(self,ep,whole_checkpoint=True,verify=False):
 import torch
 from torch.utils.checkpoint import checkpoint
 from memory_geometry_followup import geometry_loss
 state=self.model.initial_state();obs=ep['observations']
 if not ep['supports']:return torch.zeros((),device='cuda'),None,None
 for j in ep['supports']:
  z,a=self.chunk(j//32*32)
  def forward(initial,actions,current):
   with torch.autocast('cuda',dtype=torch.bfloat16):return self.model(initial,actions,current.expand(len(actions),-1,-1))
  pred=checkpoint(forward,z,a,state,use_reentrant=False) if whole_checkpoint else forward(z,a,state)
  residual=torch.from_numpy(obs[j]['latent'][None]).cuda().float()-pred[j%32:j%32+1]
  with torch.autocast('cuda',dtype=torch.bfloat16):state=self.model.update_state(state,residual,torch.from_numpy(self.pool['action'][j:j+1]).cuda())
 if verify:assert torch.equal(state.detach(),ep['final_state'])
 import numpy as np
 qi=ep['query'];z0=torch.from_numpy(self.pool['z0'][qi].copy()).cuda().float();target=torch.from_numpy(np.stack([obs[j]['latent'] for j in qi])).cuda().float();actions=torch.from_numpy(self.pool['action'][qi]).cuda()
 def terms(current):
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.model(z0,actions,current.expand(8,-1,-1)).float()
  ls=latent_terms(pred,target,self.smooth_beta);geo=geometry_loss(self.probe,pred,[obs[j]['features'] for j in qi],ep['times'])
  return self.loss_scale*ls[self.loss_kind]+.1*geo,ls['l1'].detach(),geo.detach()
 return checkpoint(terms,state,use_reentrant=False) if whole_checkpoint else terms(state)

def gradient_norm(model):
 import torch
 return float(torch.sqrt(sum(p.grad.float().square().sum() for p in model.parameters() if p.grad is not None)))

def load(team,panel,kind='l1',seed=0):
 import torch
 path=B/'models'/panel/'uniform/checkpoint_128.json' if kind=='l1' and seed==0 else R/'models'/f'seed{seed}'/panel/kind/'complete.json'
 ck=read(path);assert sha(ck['checkpoint'])==ck['checkpoint_sha256'];team.model.load_state_dict(torch.load(ck['checkpoint'],map_location='cpu',weights_only=False)['state_dict']);assert source().old.reader_hash(team.model)==ck['reader_sha256'];return ck

def configure(team,panel,kind):
 cal=read(R/f'CALIBRATION_{panel}.json');assert cal['complete'];team.loss_kind=kind;team.loss_scale=cal['scales'][kind];team.smooth_beta=cal['beta'];return cal

def close(team):team.center_hook.remove();team.truth.close()
