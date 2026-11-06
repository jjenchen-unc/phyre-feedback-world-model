"""Synthetic CPU tests of instrumentation, not experiment effectiveness."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).parent))
import diagnose as d

root=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('prefix_test',root/'phyre-feedback-world-model/phyre_feedback/memory.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        base=nn.Module();base.inp=nn.Sequential(nn.Identity(),nn.Linear(8,16))
        block=nn.Module();block.sp=nn.Module();block.sp.h=2;base.blocks=nn.ModuleList([block])
        self.model=m.PrefixPredictor(base,8,n_tokens=3,mem_steps=2,mem_grid=2).eval()
        nn.init.normal_(self.model.update_attention.out_proj.weight,std=.1)
        nn.init.normal_(self.model.update_ff[-1].weight,std=.1)
        self.state=self.model.initial_state().detach();self.r=torch.randn(1,4,16,8);self.a=torch.randn(1,3)
    def test_real_and_zero_match_actual_nonidentity_updater(self):
        for variant,r in [('real',self.r),('zero_residual',torch.zeros_like(self.r))]:
            result=d.update_parts(self.model,self.state,self.r,self.a,variant,42)[0]
            expected=self.model.update_state(self.state,r,self.a)
            self.assertTrue(torch.equal(result,expected),variant)
        self.assertFalse(torch.equal(expected,self.state))
    def test_no_action_embedding_matches_hook(self):
        result=d.update_parts(self.model,self.state,self.r,self.a,'no_action_embedding',42)[0]
        h=self.model.action_in.register_forward_hook(lambda mod,args,out:torch.zeros_like(out))
        try:expected=self.model.update_state(self.state,self.r,self.a)
        finally:h.remove()
        self.assertTrue(torch.equal(result,expected))
    def test_shuffle_preserves_content_scale_but_changes_alignment(self):
        a=d.update_parts(self.model,self.state,self.r,self.a,'real',42)
        b=d.update_parts(self.model,self.state,self.r,self.a,'shuffle_pooled_tokens',42)
        self.assertAlmostEqual(a[3]['residual_projection_with_bias']['rms'],b[3]['residual_projection_with_bias']['rms'],places=6)
        self.assertFalse(torch.equal(a[1],b[1]))
    def test_mean_decomposition_preserves_channel_means(self):
        z=torch.randn(2,3,4,8)+torch.arange(8)[None,None,None,:]
        mean=d.residual_component(z,'mean_only');center=d.residual_component(z,'centered')
        torch.testing.assert_close(mean+center,z)
        torch.testing.assert_close(center.mean((1,2)),torch.zeros(2,8),atol=1e-6,rtol=0)
        torch.testing.assert_close(mean[:,0,0],z.mean((1,2)))
        self.assertGreater(float(mean.std(-1).mean()),1.)
    def test_step_scaling_preserves_endpoints_and_is_linear(self):
        state=torch.randn(2,3,8);updated=state+torch.randn_like(state)
        self.assertTrue(d.scaled_state(state,updated,0) is state)
        self.assertTrue(d.scaled_state(state,updated,1) is updated)
        for alpha in [.25,.5]:
            torch.testing.assert_close(d.scaled_state(state,updated,alpha)-state,alpha*(updated-state))
    def test_mean_and_centered_match_projection_input_hook(self):
        for variant in ['mean_only','centered']:
            result=d.update_parts(self.model,self.state,self.r,self.a,variant,42)[0]
            h=self.model.residual_in.register_forward_pre_hook(lambda module,args:(d.residual_component(args[0],variant),))
            try:expected=self.model.update_state(self.state,self.r,self.a)
            finally:h.remove()
            self.assertTrue(torch.equal(result,expected),variant)
    def test_ranking_ties_mask_and_undefined_auc(self):
        scores=np.array([5.,3.,3.,1.]);labels=np.array([1,0,1,0],bool);tried=np.array([1,0,0,0],bool)
        r=d.ranking_metrics(scores,labels,tried,scores)
        self.assertEqual(r['next_local'],1);self.assertEqual(r['best_success_rank'],2)
        self.assertEqual(r['pair_auc'],.75);self.assertFalse(r['top1_changed'])
        self.assertIsNone(d.ranking_metrics(scores,np.zeros(4,bool),tried)['pair_auc'])

if __name__=='__main__':unittest.main()
