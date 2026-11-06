"""Post-hoc descriptive review; no model fitting or checkpoint selection.

Bootstrap intervals resample paired tasks WITHIN the observed templates.
They exclude training-seed and unseen-template uncertainty and are not adjusted
for multiple comparisons. They do not guarantee population coverage for this
selected panel or account for dependencies from the wrong-scene donor graph.
Do not interpret them as generalization guarantees or equivalence tests.
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/memory_redesign_20260928'
REPEATS = 20000


def paired_summary(values, templates):
    values, templates = np.asarray(values), np.asarray(templates)
    rng = np.random.default_rng(20260929)
    boot = np.zeros(REPEATS)
    groups = sorted(set(templates))
    for template in groups:
        group = values[templates == template]
        boot += group[rng.integers(len(group), size=(REPEATS, len(group)))].sum(1)
    boot /= len(values)
    return dict(n=len(values), mean=float(values.mean()),
                conditional_task_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(),
                positive=int((values > 1e-12).sum()), negative=int((values < -1e-12).sum()),
                template_means={t: float(values[templates == t].mean()) for t in groups},
                leave_one_template_out={t: float(values[templates != t].mean())
                                        for t in groups if len(groups) > 1})


def main():
    load = lambda p: json.loads(p.read_text())
    names = ['R0', 'R1', 'R2', 'R3', 'R4']
    closed = {'base': load(ROOT / 'results/cx_clm_base.json')}
    closed.update({v: load(OUT / f'{v}_closed_loop.json') for v in names})
    rows = {v: {r['task']: r for r in d['per_task']} for v, d in closed.items()}
    tasks = sorted(rows['base'])
    assert all(set(r) == set(tasks) for r in rows.values())
    result = {'scope': __doc__, 'bootstrap_repeats': REPEATS, 'closed_comparisons': {}, 'fixed': {}}
    for v, ref in [('R0', 'base'), ('R1', 'R0'), ('R2', 'R0'), ('R3', 'R0'),
                   ('R3', 'R1'), ('R3', 'R2'), ('R4', 'R3'), ('R4', 'base')]:
        result['closed_comparisons'][f'{v}-{ref}'] = paired_summary(
            [rows[v][t]['auccess'] - rows[ref][t]['auccess'] for t in tasks], [t[:5] for t in tasks])
    data = load(OUT / 'data/dataset.json')
    index_task = {i: t for t, d in data['tasks'].items() for i in d['support_pool'] + d['query_pool']}
    summary = load(OUT / 'SUMMARY.json')
    result['training_sampling'] = {}
    for v in names:
        counts = summary['variants'][v]['training']['counts']
        result['training_sampling'][v] = dict(
            mean_k=sum(k * counts.get(f'actual_k_{k}', 0) for k in range(5)) / counts['episodes'],
            positive_query_fraction=counts['positive_queries'] / counts['queries'])
        fixed = load(OUT / f'{v}_fixed.json')['rows']
        lookup = {(r['task'], r['requested_k'], r['condition']): r for r in fixed}
        nonempty = [r for r in fixed if r['condition'] == 'correct' and r['requested_k'] == 1 and r['actual_k']]
        f = {'nonempty_tasks': len(nonempty), 'requested_k4_actual_k': dict(Counter(
            r['actual_k'] for r in fixed if r['condition'] == 'correct' and r['requested_k'] == 4))}
        f['wrong_scene_same_template'] = sum(
            index_task[lookup[r['task'], 1, 'wrong_scene']['context'][0]][:5] == r['task'][:5]
            for r in nonempty)
        for ref in ['base', 'wrong_scene']:
            valid = [r for r in nonempty if r['rank_auc'] is not None and
                     lookup[r['task'], 1, ref]['rank_auc'] is not None]
            f[f'correct_minus_{ref}_auc'] = paired_summary(
                [r['rank_auc'] - lookup[r['task'], 1, ref]['rank_auc'] for r in valid],
                [r['task'][:5] for r in valid])
        for solved in [True, False]:
            ds, ts = [], []
            for r in nonempty:
                base = lookup[r['task'], 1, 'base']
                assert [q['index'] for q in r['per_query']] == [q['index'] for q in base['per_query']]
                delta = [a['latent_l1'] - b['latent_l1'] for a, b in zip(r['per_query'], base['per_query'])
                         if a['solved'] == solved]
                if delta:
                    ds.append(float(np.mean(delta))); ts.append(r['task'][:5])
            f['success_l1_scene_delta' if solved else 'failure_l1_scene_delta'] = paired_summary(ds, ts)
        result['fixed'][v] = f
    (OUT / 'REVIEW_METRICS.json').write_text(json.dumps(result, indent=2) + '\n')
    for key, row in result['closed_comparisons'].items():
        print(key, round(row['mean'], 4), np.round(row['conditional_task_bootstrap_95'], 4).tolist())
    for v, row in result['fixed'].items():
        r = row['correct_minus_wrong_scene_auc']
        print(v, 'correct-wrong AUC', round(r['mean'], 4), np.round(r['conditional_task_bootstrap_95'], 4).tolist())


if __name__ == '__main__':
    main()
