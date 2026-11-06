"""Numerical scorer parity, input boundaries, gradients, and replay invariants."""
import json
import numpy as np
import torch
from common import surface_dist, rule_score, MemoryPredictor
from redesign_lib import Store, load_models, readout, torch_surface, context_memory
from redesign_sampling import sample_episode
from prepare_memory_redesign import replay, partitions
from train_memory_redesign import loss_terms


def main():
    torch.set_num_threads(4); torch.manual_seed(0)
    store = Store(); base, probe, rule, pc, _ = load_models()
    ids = store.by[sorted(store.by)[0]][:4]
    z = torch.from_numpy(store.z(ids)).cuda().float()
    fs = [store.feature(i) for i in ids]
    for f in fs:
        np.testing.assert_allclose(torch_surface(torch.from_numpy(f).cuda()).cpu(), surface_dist(f), atol=1e-4)
    scores, xy, dist = readout(z, fs, probe, rule, details=True)
    # Independent legacy probe path, retaining NumPy geometry/scoring.
    kv = probe.inp(z.reshape(-1, 196, z.shape[-1])) + probe.space
    legacy = []
    for b, f in enumerate(fs):
        pf = np.repeat(f[:1], z.shape[1], axis=0).copy()
        for o in range(f.shape[1]):
            if f[0, o, 13] < .5 and f[0, o, 11] < .5:
                q = torch.from_numpy(np.repeat(f[:1, o], z.shape[1], axis=0)).cuda()
                pxy, angle = probe.read(kv[b*z.shape[1]:(b+1)*z.shape[1]], q)
                pf[:, o, :2] = pxy.detach().cpu().numpy()/256
                angle = angle.detach().cpu().numpy()
                pf[:, o, 2] = np.arctan2(angle[:, 1], angle[:, 0])/(4*np.pi) % 1
        legacy.append(rule_score(surface_dist(pf), rule))
    np.testing.assert_allclose(scores.detach().cpu(), legacy, atol=1e-4, rtol=1e-4)
    # Query future geometry must not affect the readout input.
    changed = [f.copy() for f in fs]
    for f in changed: f[1:] = 123
    torch.testing.assert_close(readout(z, changed, probe, rule), scores)
    pr = z.detach().clone().requires_grad_(True)
    s = readout(pr, fs, probe, rule); s.sum().backward()
    assert torch.isfinite(pr.grad).all() and pr.grad.abs().sum() > 0
    y = torch.tensor([True, False, True, False], device='cuda')
    lp, lr, _ = loss_terms(pr, z, y, True, s)
    assert torch.isfinite(lp + lr)
    model = MemoryPredictor(base, pc['Din'], mode='residual', mem_steps=16, mem_grid=10,
                            per_attempt=True, gate_empty=True).cuda().eval()
    # Nonzero adapter weights must still be disabled by empty context.
    for o in model.o:
        torch.nn.init.normal_(o.weight, std=.01); torch.nn.init.normal_(o.bias, std=.01)
    memory, mask = context_memory(model, store, [])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        a = torch.from_numpy(store.actions[ids]).cuda()
        torch.testing.assert_close(base(z[:, 0], a), model(z[:, 0], a, memory.expand(4,-1,-1), mask.expand(4,-1)))
    # This synthetic trace must stop at success, even though more failures exist.
    class Toy: labels = np.array([False, True, False])
    r = replay([0,1,2], np.array([3.,2.,1.]), Toy())
    assert r['failures'] == [0] and r['attempts'][-1] == {'index': 1, 'solved': True}
    data = {'train_tasks': sorted(store.by)[:8], 'tasks': {}}
    for t in data['train_tasks']:
        support, query = partitions(store, t)
        fails = [i for i in support if not store.labels[i]][:4]
        data['tasks'][t] = {'support_pool':support,'query_pool':query,
                           'base_scores':{str(i):float(i) for i in store.by[t]},
                           'base_trace':{'failures':fails},'memory_trace':{'failures':fails}}
    for step in range(100):
        episodes = [sample_episode(data, store, step, 0, v) for v in ['R0','R1','R2','R3','R4']]
        assert len({e['task'] for e in episodes}) == 1
        assert episodes[0]['query'] == episodes[1]['query']
        assert episodes[2]['query'] == episodes[3]['query'] == episodes[4]['query']
    print(json.dumps({'passed': True, 'legacy_score_max_error':float(np.max(np.abs(scores.detach().cpu().numpy()-legacy))),
                      'gradient_abs_sum':float(pr.grad.abs().sum()), 'paired_sampling_steps':100}), flush=True)


if __name__ == '__main__': main()
