"""CPU-only proposal audit. Synthetic behavior is not evidence of learned performance."""
from pathlib import Path
import copy,hashlib,json,sys
import torch
import numpy as np
root=Path(sys.argv[1]);code=root/'results/query_token_memory_20261005/code';sys.path.insert(0,str(code))
from common import Predictor
from query_token_predictor import QueryTokenPredictor

torch.set_num_threads(2);torch.manual_seed(20261005)
base=Predictor(8,4,4,16,2,4)
m=QueryTokenPredictor(base,8,n_tokens=4,d_mem=16,mem_steps=2,mem_grid=2).double().eval()
# Exercise gradients beyond the intentionally zero-initialized decoder.
with torch.no_grad():
 for layer in m.o:layer.weight.normal_(std=.01);layer.bias.normal_(std=.01)
a=copy.deepcopy(m);b=copy.deepcopy(m)
res=torch.randn(3,1,4,4,8,dtype=torch.float64);acts=torch.rand(3,1,3,dtype=torch.float64)
z0=torch.randn(2,4,8,dtype=torch.float64);qact=torch.rand(2,3,dtype=torch.float64);target=torch.randn(2,4,4,8,dtype=torch.float64)
state=a.initial_state();losses=[]
for i in range(3):
 state=a.update_state(state,res[i],acts[i]);mem,mask=a.state_memory(state,True)
 pred=a(z0,qact,mem.expand(2,-1,-1),mask.expand(2,-1));losses.append((pred[:,1:]-target[:,1:]).abs().mean()/3)
sum(losses).backward()
state=b.initial_state()
for i in range(3):
 state=b.update_state(state,res[i],acts[i]);mem,mask=b.state_memory(state,True)
 pred=b(z0,qact,mem.expand(2,-1,-1),mask.expand(2,-1));loss=(pred[:,1:]-target[:,1:]).abs().mean()/3
 loss.backward(retain_graph=i<2)
errs=[]
for (ka,pa),(kb,pb) in zip(a.named_parameters(),b.named_parameters()):
 assert ka==kb
 if pa.requires_grad:
  assert pa.grad is not None and pb.grad is not None
  errs.append(float((pa.grad-pb.grad).abs().max()))
assert max(errs)<1e-10
with torch.no_grad():
 initial=m.initial_state().clone();zero=torch.zeros_like(res[0]);changed=m.update_state(initial,zero,acts[0])
 assert not torch.equal(initial,changed)
 pbase=m.base(z0,qact)
 mem,mask=m.state_memory(initial,True);active=m(z0,qact,mem.expand(2,-1,-1),mask.expand(2,-1))
 mem,mask=m.state_memory(initial,False);inactive=m(z0,qact,mem.expand(2,-1,-1),mask.expand(2,-1))
 assert torch.equal(inactive,pbase) and not torch.equal(active,pbase)
manifest=json.loads((root/'results/memory_fresh_geometry_20261005/manifest.json').read_text())
ns=np.array([len(r['failures']) for r in manifest['rows'] if r['split']=='train'])
result={'scope':'CPU synthetic graph and structural checks; no trained token model, GPU run, or performance inference.',
 'code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in code.glob('*.py')},
 'prefix_backward_max_gradient_difference':max(errs),'prefix_backward_equivalent_to_sum':True,
 'initial_tokens_receive_gradient':bool(a.initial_tokens.grad.abs().sum()>0),
 'zero_residual_can_update_tokens':True,'active_initial_tokens_can_change_prediction_without_residual':True,
 'empty_gate_identity_after_nonzero_adapters':True,
 'training_support_availability':{'scenes':len(ns),'minimum':int(ns.min()),'median':float(np.median(ns)),'at_least_8':int((ns>=8).sum()),'at_least_19':int((ns>=19).sum())}}
weights=np.zeros(8);expected_k=0.
for n in ns:
 for requested in [1,2,4,8]:
  k=min(requested,int(n));weights[:k]+=1/(len(ns)*4*k);expected_k+=k/(len(ns)*4)
result['expected_prefix_loss_weights']={str(i+1):float(v) for i,v in enumerate(weights)}
result['expected_supervised_prefixes_per_optimizer_step']=float(expected_k)
p=root/'results/query_token_proposal_review_20261005/STRUCTURE_AUDIT.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
