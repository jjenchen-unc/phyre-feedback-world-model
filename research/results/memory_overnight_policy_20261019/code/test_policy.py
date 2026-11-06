import importlib.util,sys,unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).parent))
import policy
root=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('memory_test',root/'phyre-feedback-world-model/phyre_feedback/memory.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PolicyTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(8);base=nn.Module();base.inp=nn.Sequential(nn.Identity(),nn.Linear(8,16));block=nn.Module();block.sp=nn.Module();block.sp.h=2;base.blocks=nn.ModuleList([block]);self.model=m.PrefixPredictor(base,8,n_tokens=3,mem_steps=2,mem_grid=2).eval();nn.init.normal_(self.model.update_attention.out_proj.weight,std=.1);nn.init.normal_(self.model.update_ff[-1].weight,std=.1);self.state=self.model.initial_state().detach();self.r=torch.randn(1,4,16,8);self.a=torch.randn(1,3)
 def test_intervention_restores_hooks_and_weights(self):
  saved={k:v.clone() for k,v in self.model.state_dict().items()};before=self.model.update_state(self.state,self.r,self.a)
  for variant in ['real','centered','mean_only','step_0.25','step_0.5','feedback_only']:
   out=policy.intervention(self.model,self.state,self.r,self.a,variant)
   self.assertTrue(torch.isfinite(out).all());self.assertEqual(len(self.model.residual_in._forward_pre_hooks),0)
   self.assertTrue(torch.equal(before,self.model.update_state(self.state,self.r,self.a)))
   for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,saved[k]))
 def test_gating_and_feedback_algebra(self):
  real=self.model.update_state(self.state,self.r,self.a);zero=self.model.update_state(self.state,torch.zeros_like(self.r),self.a)
  for a in [.25,.5]:torch.testing.assert_close(policy.intervention(self.model,self.state,self.r,self.a,f'step_{a}'),self.state+a*(real-self.state))
  torch.testing.assert_close(policy.intervention(self.model,self.state,self.r,self.a,'feedback_only'),self.state+real-zero)
if __name__=='__main__':unittest.main()
