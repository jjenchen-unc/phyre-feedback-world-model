import importlib.util,json
from pathlib import Path
import torch
import train
p=train.P/'results/memory_overnight_policy_20261019/code/policy.py'
spec=importlib.util.spec_from_file_location('previous',p);previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
torch.manual_seed(1821)
for dtype in [torch.float32,torch.bfloat16]:
 z=torch.randn(2,16,100,128).to(dtype)
 assert torch.equal(train.transform(z,'centered'),previous.component(z,'centered'))
z=torch.randn(2,4,7,8,dtype=torch.float64,requires_grad=True);c=train.transform(z,'centered');offset=torch.randn(2,1,1,8,dtype=torch.float64)
assert c.mean((1,2)).abs().max()<1e-14
assert torch.allclose(c,train.transform(z+offset,'centered'),atol=1e-14,rtol=1e-14)
(c*torch.randn_like(c)).sum().backward()
assert z.grad.abs().sum()>0 and z.grad.sum((1,2)).abs().max()<1e-13
train.write(train.R/'TRANSFORM_VALIDATION.json',{'passed':True,'previous_transform_bitwise_exact':['float32','bfloat16'],'feature_offset_invariance':True,'gradient_connected':True})
print('TRANSFORM_VALIDATED')
