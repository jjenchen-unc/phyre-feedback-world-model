import copy
import unittest
import numpy as np
import torch
from phyre_feedback.models import Predictor
from phyre_feedback.memory import PrefixPredictor, replay_state
from phyre_feedback.metrics import auc

torch.set_num_threads(1)
class CoreTests(unittest.TestCase):
    def model(self, zero=False):
        torch.manual_seed(7)
        base=Predictor(8,4,4,8,1,2)
        return PrefixPredictor(base,8,n_tokens=2,mem_steps=2,mem_grid=2,
                               stage='updater',zero_residual=zero,checkpoint_blocks=False).eval()
    def test_initial_tokens_match_implicit_q0_and_initial_frame_is_preserved(self):
        m=self.model();z=torch.randn(3,4,8);a=torch.rand(3,3)
        torch.testing.assert_close(m(z,a),m(z,a,m.initial_state(3)),rtol=0,atol=0)
        torch.testing.assert_close(m(z,a)[:,0],z,rtol=0,atol=0)
    def test_identity_initialized_updater_then_only_updater_learns(self):
        m=self.model();state=m.initial_state();r=torch.randn(1,4,4,8);a=torch.rand(1,3)
        torch.testing.assert_close(m.update_state(state,r,a),state,rtol=0,atol=0)
        frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
        opt.zero_grad();m.update_state(state,r,a).square().mean().backward();opt.step()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters() if p.requires_grad))
        for n,p in m.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
    def test_zero_residual_ignores_residual_even_after_nonidentity_initialization(self):
        m=self.model(True)
        with torch.no_grad():m.update_attention.out_proj.weight.normal_()
        s=m.initial_state();a=torch.rand(1,3);r=torch.randn(1,4,4,8)
        torch.testing.assert_close(m.update_state(s,r,a),m.update_state(s,torch.zeros_like(r),a),rtol=0,atol=0)
    def test_replay_uses_new_state_for_the_next_residual_and_keeps_gradients(self):
        m=self.model()
        with torch.no_grad():m.update_attention.out_proj.weight.normal_(std=.1)
        z=torch.randn(4,4,8);a=torch.rand(4,3);truth=torch.randn(4,4,4,8)
        calls=[]
        def observed(j):calls.append(j);return truth[j:j+1]
        actual=replay_state(m,z,a,[1,3],observed,batch_size=4)
        state=m.initial_state()
        for j in [1,3]:state=m.update_state(state,truth[j:j+1]-m(z,a,state.expand(4,-1,-1))[j:j+1],a[j:j+1])
        torch.testing.assert_close(actual,state,rtol=0,atol=0)
        self.assertEqual(calls,[1,3]);actual.square().mean().backward()
        self.assertIsNotNone(m.residual_in.weight.grad)
        self.assertGreater(float(m.residual_in.weight.grad.abs().sum()),0)
    def test_auc_handles_single_class_and_ties(self):
        self.assertIsNone(auc(np.array([True,True]),np.array([1.,2.])))
        self.assertEqual(auc(np.array([False,True]),np.array([1.,1.])),.5)
if __name__=='__main__':unittest.main()
