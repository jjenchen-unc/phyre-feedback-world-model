"""Verify stage A outputs and report equal-scene, class-separated comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
from attribution_a_metrics import auc, log_score


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flatten(v, prefix + k + '.'))
        elif isinstance(v, (int, float, bool)) or v is None: out[prefix + k] = v
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    manifest = json.loads((a.root / 'manifest.json').read_text())
    tasks = [r for r in manifest['tasks'] if not a.smoke or r['task'] in manifest['smoke_tasks']]
    folder = a.root / ('smoke' if a.smoke else 'tasks'); rows = []
    for entry in tasks:
        dest = folder / entry['task'].replace(':', '_')
        raw = json.loads((dest / 'summary.json').read_text())
        assert raw['entry'] == entry and raw['base_score_exact_replay'] and raw['memory_K1_score_exact_replay']
        assert len(raw['empty_parity_models']) == 5 and raw['dense_raw_checked'] == 1024
        with np.load(dest / 'truth.npz') as truth:
            y, qm = truth['labels'], truth['query_mask']; assert qm.sum() == len(entry['query'])
            for condition, stats in raw['conditions'].items():
                if condition.startswith('true_geometry'):
                    if condition == 'true_geometry': scores = log_score(truth['distance'], raw['rule'])
                    elif condition.endswith('coarse_weighted'): scores = log_score(truth['distance'], raw['rule'], weighted=True)
                    else: scores = log_score(truth['dense_distance'], raw['rule'], weighted=True)
                else:
                    path = dest / (condition + '.npz')
                    if not path.exists():
                        assert condition.endswith('_K1') and not entry['has_nonempty_K1']
                        path = dest / 'base.npz'
                    with np.load(path) as values:
                        scores = log_score(values['distance'], raw['rule'])
                        np.testing.assert_allclose(scores, values['log_score'], rtol=1e-12, atol=1e-12)
                        for group, gm in [('all', qm), ('success', qm & y), ('failure', qm & ~y)]:
                            actual = float(values['latent_l1'][gm].mean()) if gm.any() else None
                            expected = stats[group]['latent_l1']
                            assert (actual is None and expected is None) or abs(actual - expected) < 1e-7
                aa = auc(y[qm], scores[qm]); assert aa == stats['auc']
        rows.append(raw)
    if a.smoke:
        output = {'passed': True, 'tasks': len(rows), 'scope': 'Engineering check, not selection evidence'}
        (a.root / 'smoke_check.json').write_text(json.dumps(output, indent=2) + '\n'); print(output); return
    groups = {}
    groups['all100_K1'] = [('K1', r) for r in rows]
    groups['nonempty_K1_primary'] = [('K1', r) for r in rows if r['entry']['has_nonempty_K1']]
    groups['K4_matched_K1'] = [('K1', r) for r in rows if 'K4' in r['entry']['contexts']]
    groups['K4'] = [('K4', r) for r in rows if 'K4' in r['entry']['contexts']]
    result = {'complete': True, 'audit_passed': True, 'scope': manifest['scope'], 'coverage': manifest['counts'], 'panels': {}}
    for panel, entries in groups.items():
        conditions = {}
        for level, r in entries:
            for name in ['base', 'observed_latent', 'true_geometry', 'true_geometry_coarse_weighted', 'true_geometry_dense_weighted', *manifest['models']]:
                key = name if name not in manifest['models'] else name + '_' + level
                conditions.setdefault(name, []).append({'task': r['task'], **r['conditions'][key]})
        output = {}
        for name, values in conditions.items():
            flat = [flatten(v) for v in values]
            keys = sorted(set().union(*(v.keys() for v in flat)))
            output[name] = {'n_scenes': len(values), 'n_auc_scenes': sum(v['auc'] is not None for v in values),
                            'macro': {k: mean([v.get(k) for v in flat]) for k in keys if not k.endswith('_n') and not k.startswith('n_')},
                            'metric_scene_counts': {k: sum(v.get(k) is not None for v in flat) for k in keys}, 'per_task': values}
        paired = {}
        bm = {v['task']: v for v in conditions['base']}
        for name in manifest['models']:
            changes = []
            for v in conditions[name]:
                b = bm[v['task']]
                changes.append({'task': v['task'], 'auc_delta': v['auc'] - b['auc'] if v['auc'] is not None else None,
                                'latent_l1_delta': v['all']['latent_l1'] - b['all']['latent_l1'],
                                'success_surface_mae_delta': v['success']['surface_mae'] - b['success']['surface_mae'] if v['success']['surface_mae'] is not None else None})
            paired[name] = {'mean_auc_delta': mean([v['auc_delta'] for v in changes]),
                            'mean_latent_l1_delta': mean([v['latent_l1_delta'] for v in changes]), 'per_task': changes}
        result['panels'][panel] = {'conditions': output, 'paired_memory_minus_base': paired}
    (a.root / 'SUMMARY.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Attribution stage A', '', 'Reused cross development set; same query and common base contexts. No training.', '',
             '| Model | AUC | Overall latent L1 | Success surface MAE | Failure surface MAE |', '|---|---:|---:|---:|---:|']
    primary = result['panels']['nonempty_K1_primary']['conditions']
    for name, data in primary.items():
        m = data['macro']; fmt = lambda v: 'NA' if v is None else f'{v:.5f}'
        lines.append('| '+name+' | '+' | '.join(fmt(m[k]) for k in ['auc', 'all.latent_l1', 'success.surface_mae', 'failure.surface_mae'])+' |')
    lines += ['', 'Scene/class denominators and matched K4 sensitivity: SUMMARY.json. Geometry contact metrics are threshold proxies, not simulator success.']
    (a.root / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete': True, 'coverage': manifest['counts']}), flush=True)


if __name__ == '__main__': main()
