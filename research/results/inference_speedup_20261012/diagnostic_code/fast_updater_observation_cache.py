"""Exact-scorer updater training: cache observations, recompute current residuals.

This is an optional loss implementation for a new training run. It does not start
training, modify registered jobs, change the predictor batch shape, detach BPTT,
or use inference's no-gradient scorer as the differentiable training loss.
"""
import numpy as np
import torch

from fast_inference import PreparedScorer
from memory_geometry_followup import geometry_loss
from prefix_policy_predictor import policy_rollout, replay_state


def updater_loss(model, probe, head, rule, source, episode, verify=False,
                 geometry_weight=.1, cache_observations=True):
    row, query_ids, times, cap = episode
    ids = row['candidates']
    refs = [source.refs[int(i)] for i in ids] if hasattr(source,'refs') else [(0,int(i)) for i in ids]
    z0 = torch.from_numpy(np.stack([source.lat[p][j,0] for p,j in refs])).cuda().float()
    actions = torch.from_numpy(source.actions[ids]).cuda()
    features = [source.feature(i)[:1].copy() for i in ids]
    observations = {}

    def truth(local):
        if not cache_observations or local not in observations:
            # Source targets are constant, but each policy/replay prediction is
            # recomputed under current tokens. There is no cached residual here.
            observations[local] = torch.from_numpy(source.z([ids[local]])).cuda().float()
        return observations[local]

    def observe(local):
        solved = bool(source.labels[ids[local]])
        return solved, None if solved else truth(local)

    scorer = PreparedScorer(head,rule)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        trace,supports = policy_rollout(model,z0,actions,features,scorer,observe,budget=cap+1)
    assert len({t['local'] for t in trace})==len(trace)
    assert not any(t['solved'] for t in trace[:-1])
    assert all(not source.labels[ids[j]] for j in supports)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        state = replay_state(model,z0,actions,supports,truth)
    if verify:
        assert torch.equal(state.detach().cpu(),trace[-1]['state'])
    z = torch.from_numpy(source.z(query_ids)).cuda().float()
    query_actions = torch.from_numpy(source.actions[query_ids]).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pred = model(z[:,0],query_actions,state.expand(len(query_ids),-1,-1)).float()
    latent = (pred[:,1:]-z[:,1:]).abs().mean()
    geometry = geometry_loss(probe,pred,[source.feature(i) for i in query_ids],times)
    loss = latent+geometry_weight*geometry
    assert loss.requires_grad==bool(supports)
    event = {'task':row['task'],'query':query_ids,'cap':cap,
             'supports':[ids[j] for j in supports],
             'selected':[ids[t['local']] for t in trace],
             'solved':[t['solved'] for t in trace],'used_updates':len(supports),
             'loss':float(loss.detach()),'l1':float(latent.detach()),
             'geometry':float(geometry.detach())}
    return loss,event
