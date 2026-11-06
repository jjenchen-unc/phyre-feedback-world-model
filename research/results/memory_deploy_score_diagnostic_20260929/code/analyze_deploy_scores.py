"""Describe numerical and geometric effects without replacing pilot metrics."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def auc(y, scores):
    y = np.asarray(y, bool)
    if not y.any() or y.all():
        return None
    s = np.asarray(scores)
    d = s[y, None] - s[None, ~y]
    return float(((d > 0) + .5 * (d == 0)).mean())


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    result = {'scope': 'Post-hoc fixed-input transform comparison; scene-macro means, not new primary metrics.', 'models': {}}
    score_keys = ['original_score', 'sigmoid32_score', 'stable64_score', 'log_score']
    for name in ['initial_R0', 'A_s0', 'B_s0', 'A_s1', 'B_s1']:
        raw = json.loads((root / (name + '.json')).read_text())
        assert raw['original_scores_exactly_reproduced']
        task_rows = defaultdict(list)
        for r in raw['rows']:
            obs = raw['observed'][r['task']]
            assert r['query'] == obs['query']
            y = np.array(obs['labels'], bool)
            d = np.array(r['data']['distance'])
            true_d = np.array(obs['true_distance'])
            xy = np.array(r['data']['green_xy'])
            true_xy = np.array(obs['true_green_xy'])
            values = {k + '_auc': auc(y, r['data'][k]) for k in score_keys}
            values['zero_score_fraction'] = float(np.mean(np.array(r['data']['original_score']) == 0))
            values['all_tied'] = float(len(set(r['data']['original_score'])) == 1)
            values['stable_all_tied'] = float(len(set(r['data']['log_score'])) == 1)
            for label, mask in [('success', y), ('failure', ~y)]:
                values[label + '_surface_mae'] = float(np.abs(d[mask, 1:] - true_d[mask, 1:]).mean()) if mask.any() else None
                values[label + '_green_error'] = float(np.linalg.norm(xy[mask, 1:] - true_xy[mask, 1:], axis=-1).mean()) if mask.any() else None
                values[label + '_latent_l1'] = float(np.mean(np.array(r['data']['latent_l1'])[mask])) if mask.any() else None
            task_rows[(r['task'], r['condition'])].append(values)
        conditions = {}
        for c in sorted({k[1] for k in task_rows}):
            tasks = {t: {key: mean([v[key] for v in rr]) for key in rr[0]} for (t, cc), rr in task_rows.items() if cc == c}
            conditions[c] = {'tasks': tasks, 'macro': {key: mean([v[key] for v in tasks.values()]) for key in next(iter(tasks.values()))}}
        controls = {}
        for t, obs in raw['observed'].items():
            d = np.array(obs['true_distance'], dtype=np.float64)
            margin = (raw['rule'][1] - d) / raw['rule'][2]
            logterms = -np.logaddexp(0, -margin)
            logsc = np.logaddexp.reduce(logterms, axis=-1)
            controls[t] = {'true_geometry_auc': auc(obs['labels'], logsc),
                           'observed_latent_original_auc': auc(obs['labels'], obs['observed_latents']['original_score']),
                           'observed_latent_log_auc': auc(obs['labels'], obs['observed_latents']['log_score'])}
        focus = []
        for r in raw['rows']:
            if r['task'] != '00012:076' or r['attempt'] not in [1, 4, 10]:
                continue
            obs = raw['observed'][r['task']]
            y = np.array(obs['labels'], bool)
            d = np.array(r['data']['distance'])
            td = np.array(obs['true_distance'])
            scores = np.array(r['data']['log_score'])
            focus.append({'attempt': r['attempt'], 'condition': r['condition'],
                          'original_auc': auc(y, r['data']['original_score']), 'stable_auc': auc(y, scores),
                          'stable_success_rank': int(np.flatnonzero(y[np.argsort(-scores, kind='stable')])[0])+1,
                          'score64_range': [min(r['data']['stable64_score']), max(r['data']['stable64_score'])],
                          'surface_min_range': [float(d.min()), float(d.max())],
                          'success_min_distance': d[y].min(-1).tolist(),
                          'success_true_min_distance': td[y].min(-1).tolist(),
                          'success_green_final': np.array(r['data']['green_xy'])[y, -1].tolist(),
                          'success_true_green_final': np.array(obs['true_green_xy'])[y, -1].tolist()})
        result['models'][name] = {'rule': raw['rule'], 'conditions': conditions, 'controls': controls, 'focus': focus}
        print(name, {c: {k: round(v, 5) if v is not None else None for k, v in x['macro'].items() if k.endswith('auc')} for c, x in conditions.items()})
    (root / 'ANALYSIS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
