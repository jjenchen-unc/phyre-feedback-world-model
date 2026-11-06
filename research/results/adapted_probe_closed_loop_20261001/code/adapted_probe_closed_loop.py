"""Registered factorial closed-loop evaluation using the unchanged stable evaluator."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def load(root):
    m = json.loads((root / 'manifest.json').read_text())
    assert sha(m['pack'] + '_meta.npz') == m['meta_sha256']
    for path, h in m['upstream_summary_hashes'].items():
        assert sha(path) == h
        d = json.loads(Path(path).read_text()); assert d['complete'] and d['audit_passed']
    for name, h in m['code_sha256'].items(): assert sha(root / 'code' / name) == h
    assert sha(m['predictor']) == m['predictor_sha256']
    assert sha(m['historical_base']) == m['historical_base_sha256']
    return m


def run(root, index, smoke):
    m = load(root); entries = [m['conditions'][i] for i in m['smoke_indices']] if smoke else m['conditions']
    c = entries[index]
    assert sha(c['probe']) == c['probe_sha256']
    if c['memory']: assert sha(c['memory']) == c['memory_sha256']
    if not smoke:
        check = json.loads((root / 'smoke_check.json').read_text())
        assert check['passed'] and check['manifest_sha256'] == sha(root / 'manifest.json')
    out = root / ('smoke' if smoke else 'evaluation') / (c['name'] + '.json')
    cmd = [sys.executable, '-u', str(root / 'code/closed_loop_stable_scores.py'),
           '--pack', m['pack'], '--predictor', m['predictor'], '--probe', c['probe'],
           '--score-mode', 'stable_log', '--max-updates', '1' if smoke else str(m['max_updates']),
           '--keep', str(m['keep']), '--out', str(out)]
    if c['memory']: cmd += ['--memory', c['memory'], '--feedback']
    if smoke: cmd += ['--tasks', *m['smoke_tasks']]
    subprocess.run(cmd, check=True)


def auc(y, score):
    if not y.any() or y.all(): return None
    delta = score[y, None] - score[None, ~y]
    return float(((delta > 0) + .5 * (delta == 0)).mean())


def first_rank(y, score):
    if not y.any(): return None
    return int(np.flatnonzero(y[np.argsort(-score, kind='stable')])[0]) + 1


def aggregate(rows):
    first = np.asarray([r['first_success'] for r in rows])
    return {'n_tasks': len(rows), 'auccess': float(np.mean([r['auccess'] for r in rows])),
            'success_at_1': float(np.mean(first == 1)),
            'success_at_10': float(np.mean((first > 0) & (first <= 10))),
            'success_at_100': float(np.mean(first > 0))}


def summarize(root, smoke):
    m = load(root); folder = root / ('smoke' if smoke else 'evaluation')
    conditions = [m['conditions'][i] for i in m['smoke_indices']] if smoke else m['conditions']
    expected = m['smoke_tasks'] if smoke else m['tasks']; limit = 1 if smoke else m['max_updates']
    with np.load(m['pack'] + '_meta.npz') as packed:
        labels = packed['solved'].astype(bool); actions = packed['action']; tasks = packed['tasks']
    hist = json.loads(Path(m['historical_base']).read_text()); historical = {r['task']: r for r in hist['per_task']}
    weights = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))
    results = {}; first_updates = {}; task_count = 0; attempt_count = 0
    for c in conditions:
        path = folder / (c['name'] + '.json'); data = json.loads(path.read_text())
        rows = data['per_task']; assert [r['task'] for r in rows] == expected
        expected_args = {'pack': m['pack'], 'predictor': m['predictor'], 'probe': c['probe'],
                         'memory': c['memory'], 'feedback': bool(c['memory']), 'max_updates': limit,
                         'keep': m['keep'], 'out': str(path), 'score_mode': 'stable_log'}
        for k, value in expected_args.items(): assert data['args'][k] == value
        assert data['args']['tta_steps'] == 0 and data['args']['mem_scale'] == 1.
        prov = data['provenance']
        assert prov['predictor'] == m['predictor_sha256'] and prov['probe'] == c['probe_sha256']
        assert prov['meta'] == m['meta_sha256'] and prov['code'] == m['code_sha256']['closed_loop_stable_scores.py']
        assert prov['common'] == m['code_sha256']['common.py']
        if c['memory']: assert prov['memory'] == c['memory_sha256']
        parts = Path(str(path) + '.tasks'); updates_rows = []
        for row in rows:
            task = row['task']; stem = task.replace(':', '_'); trace = json.loads((parts / (stem + '.json')).read_text())
            assert trace['result'] == row and trace['provenance'] == prov and trace['args'] == data['args']
            assert trace['score_channels'] == ['legacy_tanh32', 'stable_log64']
            with np.load(parts / trace['score_file']) as saved:
                indices = saved['indices']; y = saved['labels']; scores = saved['scores'][:, 1]
                np.testing.assert_array_equal(indices, np.flatnonzero(tasks == task))
                np.testing.assert_array_equal(y, labels[indices]); assert len(y) == 1024 and np.isfinite(scores).all()
                assert row['n_solutions_in_shortlist'] == int(y.sum()) == historical[task]['n_solutions_in_shortlist']
                # Every first score is replayed against the previously frozen base prediction + this head.
                source = Path(m['probe_evaluation']) / stem / (c['head'] + '__base.npz')
                with np.load(source) as base_scores: np.testing.assert_array_equal(scores[0], base_scores['log_score'])
                tried = np.zeros(len(y), bool); context = []; updates = 0; first = 0
                for attempt, t in enumerate(trace['attempts'], 1):
                    assert t['attempt'] == attempt and t['context'] == context
                    assert t['score_snapshot'] == updates and trace['contexts'][updates] == context
                    j = int(np.argmax(np.where(tried, -np.inf, scores[updates])))
                    assert t['candidate_local'] == j and t['pack_index'] == int(indices[j])
                    assert t['score'] == scores[updates, j] and t['solved'] == bool(y[j])
                    np.testing.assert_array_equal(t['action'], actions[indices[j]].tolist())
                    assert not tried[j]; tried[j] = True; attempt_count += 1
                    if y[j]:
                        first = attempt; assert attempt == len(trace['attempts']); break
                    if c['memory'] and updates < limit:
                        context = (context + [int(indices[j])])[-m['keep']:]; updates += 1
                        assert all(not labels[i] for i in context)
                assert len(trace['attempts']) == (first or 100)
                assert first == row['first_success'] and updates == row['updates']
                assert len(scores) == updates + 1 == len(trace['contexts'])
                assert trace['contexts'][-1] == context
                value = float((weights * (np.arange(1, 101) >= first)).sum() / weights.sum()) if first else 0.
                assert abs(row['auccess'] - value) < 1e-12
                if c['name'] == 'base_original':
                    assert first == historical[task]['first_success'] and abs(value - historical[task]['auccess']) < 1e-12
                if c['memory']:
                    base_path = folder / (f"base_p{c['probe_seed']}.json.tasks")
                    bt = json.loads((base_path / (stem + '.json')).read_text())
                    assert trace['attempts'][0]['pack_index'] == bt['attempts'][0]['pack_index']
                    with np.load(base_path / bt['score_file']) as b: np.testing.assert_array_equal(scores[0], b['scores'][0, 1])
                    assert (first == 1) == (bt['result']['first_success'] == 1)
                    if len(scores) > 1:
                        j = trace['attempts'][0]['candidate_local']; assert not y[j]
                        mask = np.arange(len(y)) != j; yy = y[mask]
                        before, after = first_rank(yy, scores[0, mask]), first_rank(yy, scores[1, mask])
                        a0, a1 = auc(yy, scores[0, mask]), auc(yy, scores[1, mask])
                        updates_rows.append({'task': task, 'first_failure': int(indices[j]), 'has_solution': bool(yy.any()),
                            'base_first_success': bt['result']['first_success'], 'memory_first_success': first,
                            'rank_before': before, 'rank_after': after,
                            'rank_delta': after - before if before is not None else None,
                            'auc_before': a0, 'auc_after': a1, 'auc_delta': a1 - a0 if a0 is not None else None})
                task_count += 1
        agg = aggregate(rows)
        assert round(agg['auccess'], 4) == data['aggregate']['auccess']
        for k in [1, 10, 100]: assert round(agg[f'success_at_{k}'], 3) == data['aggregate']['success_at'][str(k)]
        results[c['name']] = {**c, **agg, 'by_template': {t: aggregate([r for r in rows if r['task'].startswith(t)]) for t in sorted({r['task'][:5] for r in rows})}, 'per_task': rows}
        if c['memory']:
            good = [r for r in updates_rows if r['rank_delta'] is not None]
            good_auc = [r for r in updates_rows if r['auc_delta'] is not None]
            first_updates[c['name']] = {'n_first_failures': len(updates_rows), 'n_with_solution': len(good),
                'rank_improved': sum(r['rank_delta'] < 0 for r in good),
                'rank_unchanged': sum(r['rank_delta'] == 0 for r in good),
                'rank_worsened': sum(r['rank_delta'] > 0 for r in good),
                'n_auc_defined': len(good_auc),
                'mean_auc_delta': float(np.mean([r['auc_delta'] for r in good_auc])) if good_auc else None,
                'per_task': updates_rows}
    if smoke:
        assert all(d['n_first_failures'] == len(expected) for d in first_updates.values())
        save(root / 'smoke_check.json', {'passed': True, 'manifest_sha256': sha(root / 'manifest.json'),
             'conditions': len(conditions), 'tasks_checked': task_count, 'attempts_replayed': attempt_count,
             'historical_base_exact': True, 'all_initial_scores_exact': True,
             'scope': 'Two fixed scenes, max one memory update; engineering only, not model selection.'})
        print('SMOKE_PASSED', task_count, attempt_count, flush=True); return
    comparisons = {}
    metrics = ['auccess', 'success_at_1', 'success_at_10', 'success_at_100']
    def paired(a, b):
        aa, bb = results[a], results[b]
        rr = [{'task': x['task'], 'auccess_delta': x['auccess'] - y['auccess'],
               'first_success_a': x['first_success'], 'first_success_b': y['first_success']}
              for x, y in zip(aa['per_task'], bb['per_task'])]
        return {'a': a, 'b': b, 'delta': {k: aa[k] - bb[k] for k in metrics},
                'auccess_improved_tasks': sum(r['auccess_delta'] > 0 for r in rr),
                'auccess_worsened_tasks': sum(r['auccess_delta'] < 0 for r in rr),
                'by_template': {t: {k: aa['by_template'][t][k] - bb['by_template'][t][k] for k in metrics} for t in aa['by_template']}, 'per_task': rr}
    for ps in range(3):
        comparisons[f'probe_p{ps}'] = paired(f'base_p{ps}', 'base_original')
        for ms in range(3):
            for arm in ['latent_only', 'latent_geometry']:
                name = f'{arm}_m{ms}_p{ps}'; comparisons[name + '_vs_base'] = paired(name, f'base_p{ps}')
            comparisons[f'geometry_vs_l1_m{ms}_p{ps}'] = paired(f'latent_geometry_m{ms}_p{ps}', f'latent_only_m{ms}_p{ps}')
    groups = {}
    for group in ['base_original', 'base_adapted', 'latent_only', 'latent_geometry']:
        values = [r for r in results.values() if r['group'] == group]
        groups[group] = {'n_configurations': len(values), **{key: {'mean': float(np.mean([v[key] for v in values])),
            'min': float(min(v[key] for v in values)), 'max': float(max(v[key] for v in values))} for key in metrics}}
    save(root / 'SUMMARY.json', {'complete': True, 'trace_audit_passed': True, 'manifest_sha256': sha(root / 'manifest.json'),
         'scope': m['scope'], 'tasks_checked': task_count, 'attempts_replayed': attempt_count,
         'historical_base_exact': True, 'all_initial_scores_exact': True,
         'groups': groups, 'conditions': results, 'paired_comparisons': comparisons, 'first_update': first_updates})
    lines = ['# Adapted-probe closed-loop results', '', m['scope'], '',
             '| Condition | AUCCESS | S@10 | S@100 |', '|---|---:|---:|---:|']
    for name, r in results.items(): lines.append(f"| {name} | {r['auccess']:.5f} | {r['success_at_10']:.1%} | {r['success_at_100']:.1%} |")
    lines += ['', 'All 22 conditions retained. Nine memory/probe combinations share pretrained models; not nine independent full-training seeds.',
              'First-update diagnostics exclude each head-specific first failed action; denominator may differ across probe seeds.',
              'Cached simulator trajectories are revealed only after failed attempts. No new rollout collection or new holdout test.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'complete': True, 'tasks_checked': task_count, 'attempts_replayed': attempt_count, 'groups': groups}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['run', 'summary'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--index', type=int, default=0)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    if a.stage == 'run': run(a.root, a.index, a.smoke)
    else: summarize(a.root, a.smoke)


if __name__ == '__main__': main()
