"""Post-hoc scorer audit; preserves original forwards and primary metrics."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deploy_pilot_lib import PilotStore
from redesign_lib import load_models, context_memory, readout, torch_surface, digest, write_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    raw = json.loads((a.root / (a.name + '_fixed.json')).read_text())
    assert digest(raw['memory']) == raw['memory_sha256']
    store = PilotStore()
    base, probe, rule, _, model = load_models(memory=raw['memory'])
    cache, observed = {}, {}
    rows = []
    start = time.time()

    def evaluate(q, ctx, condition):
        mem = context_memory(model, store, ctx) if condition != 'base' else None
        data = {k: [] for k in ['original_score', 'sigmoid32_score', 'stable64_score', 'log_score', 'distance', 'green_xy', 'latent_l1']}
        for i in range(0, len(q), 16):
            ids = q[i:i+16]
            z = torch.from_numpy(store.z(ids)).cuda().float()
            act = torch.from_numpy(store.actions[ids]).cuda()
            if condition == 'observed':
                pred = z
            else:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = base(z[:, 0], act) if condition == 'base' else model(z[:, 0], act, mem[0].expand(len(ids), -1, -1), mem[1].expand(len(ids), -1))
                pred = pred.float()
            sc, xy, d = readout(pred, [store.feature(j) for j in ids], probe, rule, details=True)
            # Same float32 geometry, only the final scalar transform changes.
            margin32 = (rule[1] - d) / rule[2]
            margin64 = (float(rule[1]) - d.double()) / float(rule[2])
            vals = {'original_score': sc, 'sigmoid32_score': torch.sigmoid(margin32).sum(-1),
                    'stable64_score': torch.sigmoid(margin64).sum(-1),
                    'log_score': torch.logsumexp(F.logsigmoid(margin64), -1),
                    'distance': d, 'green_xy': xy,
                    'latent_l1': (pred[:, 1:] - z[:, 1:]).abs().mean((1, 2, 3))}
            for key, value in vals.items():
                data[key].extend(value.cpu().tolist())
        return data

    for r in raw['rows']:
        if r['condition'] == 'empty':
            continue
        key = (r['task'], r['condition'], tuple(r['context']), tuple(r['query']))
        if key not in cache:
            cache[key] = evaluate(r['query'], r['context'], r['condition'])
        v = cache[key]
        old = [x['score'] for x in r['per_query']]
        np.testing.assert_allclose(v['original_score'], old, atol=0, rtol=0)
        rows.append({k: r[k] for k in ['task', 'attempt', 'actual_k', 'condition', 'context', 'query']} | {'data': v})
        if r['task'] not in observed:
            q = r['query']
            vobs = evaluate(q, [], 'observed')
            truth_d, truth_xy = [], []
            for j in q:
                f = torch.as_tensor(store.feature(j), device='cuda')
                truth_d.append(torch_surface(f).cpu().tolist())
                g = int(torch.nonzero(f[0, :, 9] > .5)[0, 0])
                truth_xy.append((f[:, g, :2] * 256).cpu().tolist())
            observed[r['task']] = {'query': q, 'labels': store.labels[q].tolist(),
                                    'observed_latents': vobs, 'true_distance': truth_d, 'true_green_xy': truth_xy}
            print(json.dumps({'model': a.name, 'task': r['task'], 'seconds': round(time.time()-start)}), flush=True)
    write_json(a.out / (a.name + '.json'), {'model': a.name, 'memory_sha256': raw['memory_sha256'],
               'input_sha256': digest(a.root / (a.name + '_fixed.json')), 'rule': rule,
               'original_scores_exactly_reproduced': True, 'rows': rows, 'observed': observed,
               'seconds': round(time.time()-start), 'unique_forwards': len(cache)})
    print(json.dumps({'complete': a.name, 'rows': len(rows), 'seconds': round(time.time()-start)}), flush=True)


if __name__ == '__main__':
    main()
