"""CPU-only post-hoc analysis of saved seed-0 cross scores; no new inference."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
HEAD = 'predicted_mix_s0'

def read(p):
    return json.loads(p.read_text())

def auc(y, s):
    pos, neg = s[y], np.sort(s[~y])
    if not len(pos) or not len(neg):
        return None
    return float((np.searchsorted(neg, pos, 'left') + np.searchsorted(neg, pos, 'right')).sum() / (2 * len(pos) * len(neg)))

def panel(p):
    d = read(p / 'summary.json')
    with np.load(p / 'truth.npz') as z:
        mask = z['query_mask'].astype(bool)
        y = z['labels'][mask].astype(bool)
    result = {'task': d['task'], 'support': len(d['context'])}
    for kind in ['base', 'memory']:
        with np.load(p / (HEAD + '__' + kind + '.npz')) as z:
            # Scorer arrays use the same field name as the maintained metric code.
            scores = z['log_score'][mask]
        a = auc(y, scores)
        expected = d['conditions'][HEAD + '__' + kind]['auc']
        assert a is None and expected is None or a is not None and abs(a - expected) < 1e-12
        result[kind] = a
    return result

def first(y, scores):
    order = np.argsort(-scores, kind='stable')
    hits = np.flatnonzero(y[order])
    return int(hits[0]) + 1 if len(hits) else 0

out = {'scope': 'Post-hoc seed0 only; observational diagnosis, not new training or held-out confirmation', 'models': {}}
for arm in ['A_frozen_memory_s0', 'B_pretrained_memory_s0']:
    root = ROOT / 'evaluation' / arm
    fixed = [panel(p.parent) for p in sorted((root / 'fixed').glob('*/summary.json'))]
    valid = [r for r in fixed if r['support'] and r['base'] is not None]
    delta = np.array([r['memory'] - r['base'] for r in valid])
    boot = np.random.default_rng(20261005).choice(delta, (10000, len(delta))).mean(1)
    fixed_summary = {'n': len(valid), 'delta_auc': float(delta.mean()),
        'scene_bootstrap_95_conditional_on_templates_and_model': np.quantile(boot, [.025, .975]).tolist(),
        'improved': int((delta > 0).sum()), 'worsened': int((delta < 0).sum()), 'tied': int((delta == 0).sum())}
    diagnostics = defaultdict(dict)
    for p in sorted((root / 'diagnostics').glob('*/*/*/summary.json')):
        row = panel(p.parent)
        diagnostics[(row['task'], row['support'])][p.parent.name] = row
    ds = {}
    for k in [1, 2, 4, 8]:
        rr = [v for (task, kk), v in diagnostics.items() if kk == k and v['full']['base'] is not None]
        ds[str(k)] = {'n': len(rr), 'base': float(np.mean([v['full']['base'] for v in rr])) if rr else None,
            **{variant: float(np.mean([v[variant]['memory'] for v in rr])) if rr else None for variant in ['full', 'latest', 'zero', 'static']}}
    cases = []
    for p in sorted((root / 'closed' / 'full').glob('*/trace.json')):
        d = read(p)
        with np.load(p.parent / 'snapshots.npz') as z:
            y = z['labels'].astype(bool)
            base_rank = first(y, z['base_scores'])
            initial_rank = first(y, z['scores'][0])
            ranks = []
            tried = np.zeros(len(y), bool)
            for i, a in enumerate(d['attempts']):
                ranks.append(first(y[~tried], z['scores'][i][~tried]))
                tried[a['local']] = True
        assert d['result']['base_first_success'] == (base_rank if 0 < base_rank <= 20 else 0)
        cases.append({'task': d['result']['task'], 'base_rank': base_rank,
            'trained_backbone_no_update_rank': initial_rank,
            'full_first_success': d['result']['first_success'],
            'best_success_rank_among_untried_each_step': ranks})
    counts = {}
    for label, key in [('base', 'base_rank'), ('trained_backbone_no_update', 'trained_backbone_no_update_rank'), ('full', 'full_first_success')]:
        counts[label] = {str(k): sum(0 < r[key] <= k for r in cases) for k in [1, 9, 20]}
    out['models'][arm] = {'fixed_summary': fixed_summary, 'fixed_per_scene': fixed,
        'diagnostics_common_within_each_k': ds, 'closed_counts': counts, 'closed_cases': cases}
path = ROOT / 'INTERIM_FAILURE_ANALYSIS.json'
path.write_text(json.dumps(out, indent=2) + '\n')
for name, data in out['models'].items():
    print(name, json.dumps({k: v for k, v in data.items() if k not in ['fixed_per_scene', 'closed_cases']}))
    print('case_00000:050', next(v for v in data['closed_cases'] if v['task'] == '00000:050'))
