"""Read-only audit of existing frozen support pilot artifacts; no model forward."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path('/path/to/phyre-unc')
RUN = ROOT / 'results/support_utility_20261004'
OUT = ROOT / 'results/independent_memory_review_20261004/DATA_FINDINGS.json'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

m = read(RUN / 'manifest.json')
rows = [r for r in m['rows'] if r['split'] == 'train']
sources = {str(RUN / 'manifest.json'): sha(RUN / 'manifest.json')}
train = {}
for p in sorted((RUN / 'models').glob('*/complete.json')):
    d = read(p)
    dp = p.parent / 'draws.json'
    draws = read(dp)
    assert d['complete'] and d['steps'] == len(draws) == 2000
    sources[str(p)] = sha(p)
    sources[str(dp)] = sha(dp)
    train[p.parent.name] = {
        k: d[k] for k in ['unique_supports', 'no_success_query_steps',
                          'support_categories', 'common_draw_sha256']}
    train[p.parent.name].update(
        actual_query_success_fraction=sum(r['query_successes'] for r in draws) / sum(len(r['query']) for r in draws),
        unique_queries=len({i for r in draws for i in r['query']}),
        unique_scenes=len({r['task'] for r in draws}))
for seed in range(3):
    assert len({train[f'{arm}_s{seed}']['common_draw_sha256'] for arm in m['arms']}) == 1
natural = []
for row in m['cross_rows']:
    if not row['natural_context']:
        continue
    p = RUN / 'evaluation/uniform_s0/cross_natural' / row['task'].replace(':', '_') / 'natural/truth.npz'
    sources[str(p)] = sha(p)
    with np.load(p) as d:
        i = int(np.flatnonzero(d['indices'] == row['natural_context'][0])[0])
        assert not d['labels'][i]
        minimum = float(d['distance'][i, 1:].min())
        initial = float(d['distance'][i, 0])
    natural.append(dict(task=row['task'], index=row['natural_context'][0], minimum=minimum,
        initial=initial, category='N' if minimum <= 5 else 'far' if minimum >= 20 else 'middle'))
out = {
    'scope': 'Existing training draws and saved simulator geometry only. No model training or new inference.',
    'source_sha256': sources,
    'auditor_sha256': sha(Path(__file__)),
    'training': train,
    'common_training': {
        'scenes': len(rows),
        'template_counts': dict(Counter(r['task'][:5] for r in rows)),
        'N_support_count': sum(len(r['N']) for r in rows),
        'N_initial_near': sum(r['quality'][str(i)]['initial'] <= 5 for r in rows for i in r['N']),
        'N_per_scene_percentiles_0_25_50_75_100': np.percentile([len(r['N']) for r in rows], [0,25,50,75,100]).tolist()},
    'natural_support_counts': dict(Counter(r['category'] for r in natural)),
    'natural_support_by_template': {t: dict(Counter(r['category'] for r in natural if r['task'][:5] == t))
        for t in sorted({r['task'][:5] for r in natural})},
    'natural_support_rows': natural,
    'limits': ['far is geometry >=20px, not a claim about H membership under support-partition percentile',
               'Saved geometric distances reused; original PHYRE simulator not rerun by this audit',
               'Inventory quality reused for train N initial distance; covered by previous independent DATA_AUDIT']}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ['common_training','natural_support_counts','natural_support_by_template']}))
