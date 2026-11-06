"""CPU saved-output audit plus a small local synthetic CUDA transform check."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = {'scope': 'Saved-score audit plus synthetic transform test; no predicted-distance replay yet.', 'models': {}}
    for name in ['initial_R0', 'A_s0', 'B_s0', 'A_s1', 'B_s1']:
        raw = json.loads((a.root / (name + '_fixed.json')).read_text())
        groups = collections.defaultdict(list)
        for r in raw['rows']:
            if r['condition'] != 'empty':
                groups[r['condition']].append(r)
        result['models'][name] = {}
        for c, rows in groups.items():
            zeros = sum(q['score'] == 0 for r in rows for q in r['per_query'])
            ties = collections.Counter(r['task'] for r in rows if len(set(q['score'] for q in r['per_query'])) == 1)
            m = {'query_prefix_records': sum(len(r['per_query']) for r in rows), 'zero_score_records': zeros,
                 'all_tied_prefixes_by_task': dict(ties), 'per_task': {}}
            for task in sorted(set(r['task'] for r in rows)):
                qs = [q for r in rows if r['task'] == task for q in r['per_query']]
                m['per_task'][task] = {str(y): float(np.mean([q['latent_l1'] for q in qs if q['solved'] == y]))
                                      if any(q['solved'] == y for q in qs) else None for y in [False, True]}
            result['models'][name][c] = m
    x = torch.linspace(20, 24, 400001, device='cuda', dtype=torch.float32)
    old = .5 * (1 + torch.tanh((4 - x) / 2))
    stable = torch.sigmoid(4 - x)
    z = torch.nonzero(old == 0)[0, 0].item()
    result['synthetic_transform'] = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                                     'rule': ['soft', 4, 1], 'first_zero_distance_on_grid': x[z].item(),
                                     'stable_sigmoid_at_same_distance': stable[z].item(), 'cases': []}
    for d in [20., 22., 23., 30., 40.]:
        value = torch.tensor(d, device='cuda', dtype=torch.float32)
        result['synthetic_transform']['cases'].append({'distance': d,
            'tanh32': (.5 * (1 + torch.tanh((4 - value) / 2))).item(), 'sigmoid32': torch.sigmoid(4 - value).item()})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
