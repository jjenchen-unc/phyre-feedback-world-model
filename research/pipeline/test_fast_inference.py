"""Numerical, causal, masking and cache tests for exact inference (CPU or CUDA)."""
import argparse
import unittest

import numpy as np
import torch

from common import AttnProbe, Predictor, surface_dist
from attribution_a_metrics import log_score
from prefix_policy_predictor import PrefixPredictor
from fast_inference import CandidateRanker, PreparedProbeReadout, PreparedScorer


def reference(probe, pred, features, rule):
    # Same arithmetic and per-object shape as probe_adaptation.readout; .to(device)
    # replaces only its hard-coded .cuda() to allow CPU structural tests.
    b, nt, ns, _ = pred.shape
    with torch.no_grad(), torch.autocast(pred.device.type, enabled=False):
        kv = probe.inp(pred.float().reshape(b*nt, ns, -1)) + probe.space
        distances = []
        for j, f in enumerate(features):
            pf = np.repeat(f[:1], nt, 0).copy()
            for obj in range(f.shape[1]):
                if f[0, obj, 13] >= .5 or f[0, obj, 11] >= .5:
                    continue
                q = torch.from_numpy(np.repeat(f[:1, obj], nt, 0)).to(pred.device)
                xy, angle = probe.read(kv[j*nt:(j+1)*nt], q)
                pf[:, obj, :2] = xy.cpu().numpy()/256
                angle = angle.cpu().numpy()
                pf[:, obj, 2] = (np.arctan2(angle[:, 1], angle[:, 0])/2)/(2*np.pi)%1
            distances.append(surface_dist(pf))
        return log_score(np.stack(distances), rule)


DEVICE = 'cpu'


class ExactInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(912)
        self.device = torch.device(DEVICE)
        self.probe = AttnProbe(8, 4, d=16).to(self.device).eval().requires_grad_(False)
        self.model = Predictor(8, 4, 4, 16, 2, 4).to(self.device).eval()
        self.features = []
        for i in range(35):
            f = np.zeros((4, 4, 14), np.float32)
            f[..., :2] = [.3, .6]; f[..., 3] = .12; f[..., 4] = 1
            f[:, 0, 9] = 1                 # dynamic green ball
            f[:, 1, 10] = 1                # dynamic blue bar
            f[:, 1, 4] = 0; f[:, 1, 5] = 1
            f[:, 2, 11] = 1                # static purple target
            f[:, 3, 13] = 1                # static black object
            f[:, 1, 0] += i*.001
            self.features.append(f)
        self.z0 = torch.randn(35, 4, 8, device=self.device)
        self.actions = torch.rand(35, 3, device=self.device)
        self.rule = ('soft', 2., .5)

    def predict(self, model, z, a, state=None):
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16,
                                            enabled=self.device.type == 'cuda'):
            return model(z, a) if state is None else model(z, a, state)

    def test_reference_exact_and_partial_chunk(self):
        ranker = CandidateRanker(self.model, self.probe, self.rule,
                                 self.z0, self.actions, self.features)
        result = ranker.rank()
        expected = []
        for start in [0, 32]:
            pred = self.predict(self.model, self.z0[start:start+32], self.actions[start:start+32])
            expected.extend(reference(self.probe, pred, self.features[start:start+32], self.rule))
        np.testing.assert_array_equal(result.scores, expected)
        self.assertEqual(result.selected, int(np.argmax(expected)))
        self.assertEqual(set(ranker.layouts), {0, 32})

    def test_stable_ties_mask_and_exhaustion(self):
        z = self.z0[:1].repeat(35, 1, 1); a = self.actions[:1].repeat(35, 1)
        ranker = CandidateRanker(self.model, self.probe, self.rule, z, a,
                                 [self.features[0]]*35)
        # Restrict to the full B32 chunk so the last partial predictor shape does
        # not itself introduce a cross-batch rounding difference in this tie test.
        tried = np.zeros(35, bool); tried[32:] = True
        first = ranker.rank(tried=tried)
        np.testing.assert_array_equal(first.scores[:32], np.repeat(first.scores[0], 32))
        self.assertEqual(first.selected, 0)
        tried[:3] = True
        self.assertEqual(ranker.rank(tried=tried).selected, 3)
        tried[:] = True
        last = ranker.rank(tried=tried)
        self.assertIsNone(last.selected); self.assertIsNone(last.prediction)

    def test_future_is_not_a_ranking_input(self):
        changed = [f.copy() for f in self.features]
        for f in changed: f[1:] = np.random.default_rng(4).normal(size=f[1:].shape)
        a = CandidateRanker(self.model, self.probe, self.rule, self.z0, self.actions, self.features)
        b = CandidateRanker(self.model, self.probe, self.rule, self.z0, self.actions, changed)
        np.testing.assert_array_equal(a.rank().scores, b.rank().scores)

    def test_no_stale_weights_or_memory(self):
        model = PrefixPredictor(self.model, 8, n_tokens=3, mem_steps=2,
                                mem_grid=2, checkpoint_blocks=False).to(self.device).eval()
        ranker = CandidateRanker(model, self.probe, self.rule, self.z0,
                                 self.actions, self.features)
        state = model.initial_state().detach().clone()
        before = ranker.rank(state).scores
        other = state + torch.randn_like(state)
        result = ranker.rank(other)
        expected = []
        for start in [0,32]:
            end = min(start+32, 35)
            pred = self.predict(model, self.z0[start:end], self.actions[start:end],
                                other.expand(end-start, -1, -1))
            expected.extend(reference(self.probe, pred, self.features[start:end], self.rule))
        np.testing.assert_array_equal(result.scores, expected)
        self.assertFalse(np.array_equal(before, result.scores))
        with torch.no_grad(): model.base.out[1].bias.add_(.7)
        updated = ranker.rank(other).scores
        self.assertFalse(np.array_equal(updated, result.scores))

    def test_static_only_and_reject_training_mode(self):
        features = [f.copy() for f in self.features[:2]]
        for f in features: f[...,13] = 1
        pred = self.predict(self.model, self.z0[:2], self.actions[:2])
        actual = PreparedProbeReadout(features, 4, self.device)(self.probe, pred, self.rule)
        np.testing.assert_array_equal(actual, reference(self.probe, pred, features, self.rule))
        self.model.train()
        with self.assertRaises(ValueError):
            CandidateRanker(self.model,self.probe,self.rule,self.z0,self.actions,self.features)

    def test_scorer_cache_tracks_content_and_is_bounded(self):
        scorer = PreparedScorer(self.probe, self.rule, max_layouts=1)
        features = [f.copy() for f in self.features[:2]]
        pred = self.predict(self.model, self.z0[:2], self.actions[:2])
        first = scorer(pred, features).cpu().numpy()
        np.testing.assert_array_equal(first, reference(self.probe,pred,features,self.rule))
        key = next(iter(scorer.layouts))
        features[0][1:] += 9  # unobserved future must not invalidate layout
        np.testing.assert_array_equal(first, scorer(pred, features).cpu().numpy())
        self.assertEqual(key, next(iter(scorer.layouts)))
        features[0][0, 2, 0] += .4  # same object identity, changed initial scene
        actual = scorer(pred, features).cpu().numpy()
        np.testing.assert_array_equal(actual, reference(self.probe,pred,features,self.rule))
        self.assertNotEqual(key, next(iter(scorer.layouts)))
        self.assertEqual(len(scorer.layouts), 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args, remaining = parser.parse_known_args()
    DEVICE = args.device
    torch.set_num_threads(2)
    unittest.main(argv=['test_fast_inference'] + remaining)
