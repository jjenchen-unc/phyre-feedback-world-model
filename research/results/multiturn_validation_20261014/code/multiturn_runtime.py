"""New multi-turn primitives; no mutation of earlier registered experiments."""
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from fast_inference import PreparedScorer
from memory_geometry_followup import geometry_loss
from prefix_policy_predictor import policy_rollout


def query_terms(model, state, z, actions, probe, features, times, weight):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(z[:, 0], actions, state.expand(len(z), -1, -1)).float()
    l1 = (pred[:, 1:] - z[:, 1:]).abs().mean()
    geom = geometry_loss(probe, pred, features, times)
    return l1 + weight * geom, l1.detach(), geom.detach()


def loss_all_updates(model, probe, head, rule, source, episode, reference_scorer,
                     fast=True, weight=.03, verify=False):
    """Collect own-policy supports, replay full BPTT, supervise EVERY updated state.

    Whole-call checkpointing bounds stored candidate/query activations without
    truncating the state graph. Both reference and fast use identical checkpointing.
    Failed-only stress episodes belong to the validation caller, never this sampler.
    """
    row, qi, times, cap = episode
    ids = row['candidates']
    assert not set(ids) & set(qi)
    refs = [source.refs[int(i)] for i in ids] if hasattr(source, 'refs') else [(0, int(i)) for i in ids]
    z0 = torch.from_numpy(np.stack([source.lat[p][j, 0] for p, j in refs])).cuda().float()
    actions = torch.from_numpy(source.actions[ids]).cuda()
    features = [source.feature(i)[:1].copy() for i in ids]

    def truth(local):
        return torch.from_numpy(source.z([ids[local]])).cuda().float()

    scorer = PreparedScorer(head, rule) if fast else reference_scorer
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        trace, supports = policy_rollout(
            model, z0, actions, features, scorer,
            lambda j: (bool(source.labels[ids[j]]), truth(j)), budget=cap+1)
    torch.cuda.synchronize(); collection_s = time.perf_counter() - start
    assert len(set(t['local'] for t in trace)) == len(trace)
    assert all(not source.labels[ids[j]] for j in supports)
    z = torch.from_numpy(source.z(qi)).cuda().float()
    qa = torch.from_numpy(source.actions[qi]).cuda()
    qf = [source.feature(i) for i in qi]
    state = model.initial_state()
    losses, diagnostics = [], []
    for chosen in supports:
        start_chunk = chosen // 32 * 32
        end = min(start_chunk + 32, len(ids))
        # Explicit arguments prevent checkpoint recomputation closing over a later
        # loop iteration's action/chunk/state.
        def predict(a, b, c):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model(a, b, c.expand(len(a), -1, -1))
        pred = checkpoint(predict, z0[start_chunk:end], actions[start_chunk:end], state,
                          use_reentrant=False)
        residual = truth(chosen) - pred[chosen-start_chunk:chosen-start_chunk+1]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            state = model.update_state(state, residual, actions[chosen:chosen+1])

        def qloss(current):
            return query_terms(model, current, z, qa, probe, qf, times, weight)
        loss, l1, geom = checkpoint(qloss, state, use_reentrant=False)
        losses.append(loss)
        diagnostics.append([float(l1), float(geom)])
    if verify:
        assert torch.equal(state.detach().cpu(), trace[-1]['state']), 'Replay state differs'
    if losses:
        loss = torch.stack(losses).mean()
    else:
        # Preserve empty episodes in weighting and logging, without introducing
        # a gradient path or forcing additional failures.
        with torch.no_grad():
            loss, l1, geom = query_terms(model, state, z, qa, probe, qf, times, weight)
        diagnostics.append([float(l1), float(geom)])
    event = {'task': row['task'], 'cap': cap, 'queries': qi,
             'selected': [ids[t['local']] for t in trace],
             'supports': [ids[j] for j in supports],
             'solved': [t['solved'] for t in trace], 'used_updates': len(supports),
             'state_terms': diagnostics, 'loss': float(loss.detach()),
             'candidate_count': len(ids)}
    return loss, event, {'collection_s': collection_s}


class PredictorAdapter:
    """Same predictor-only adaptation for a bare base or a Q0 wrapper.

    Initial tokens remain frozen. Each scene resets weights/replay; each call
    recreates Adam. No online access to untried actions' true trajectories.
    """
    def __init__(self, model, lr=3e-4, steps=1, replay=5):
        self.model, self.lr, self.steps, self.replay = model, lr, steps, replay
        model.eval().requires_grad_(False)
        backbone = model.base if hasattr(model, 'base') else model
        for layer in (backbone.blocks[-1], backbone.out[0]):
            layer.requires_grad_(True)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.snapshot = [p.detach().clone() for p in self.params]
        self.buffer = []

    @torch.no_grad()
    def reset(self):
        for p, initial in zip(self.params, self.snapshot):
            p.copy_(initial); p.grad = None
        self.buffer = []
        self.model.eval()

    def update(self, index, z0, action, truth):
        self.buffer.append((int(index), z0.detach(), action.detach(), truth.detach()))
        self.buffer = self.buffer[-self.replay:]
        z = torch.cat([x[1] for x in self.buffer])
        a = torch.cat([x[2] for x in self.buffer])
        target = torch.cat([x[3] for x in self.buffer])
        optimizer = torch.optim.Adam(self.params, lr=self.lr)
        losses = []
        with torch.enable_grad():
            for _ in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = self.model(z, a).float()
                loss = (pred[:, 1:] - target[:, 1:]).abs().mean()
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None for p in self.model.parameters() if not p.requires_grad)
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.params)
                assert all(torch.isfinite(p.grad).all() for p in self.params if p.grad is not None)
                optimizer.step(); losses.append(float(loss.detach()))
        return {'replay_indices': [x[0] for x in self.buffer], 'losses': losses}
