"""Audit completed configurations only; never publish a partial run as final."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    m = json.loads((a.root / 'manifest.json').read_text())
    assert digest(m['pack'] + '_meta.npz') == m['meta_sha256']
    for name, h in m['code_sha256'].items(): assert digest(a.root / 'code' / name) == h
    with np.load(m['pack'] + '_meta.npz') as z:
        tasks, labels, actions = z['tasks'], z['solved'].astype(bool), z['action']
    old = {r['task']: r for r in json.loads(Path(m['historical_base']).read_text())['per_task']}
    completed = {}; attempts_checked = 0; files = {}
    def track(path):
        files[str(path.relative_to(a.root))] = digest(path)
    for c in m['conditions']:
        path = a.root / 'evaluation' / (c['name'] + '.json')
        if not path.exists(): continue
        d = json.loads(path.read_text()); track(path)
        assert len(d['per_task']) == 100 and [r['task'] for r in d['per_task']] == m['tasks']
        for field, val in [('pack', m['pack']), ('predictor', m['predictor']), ('probe', c['probe']),
                           ('memory', c['memory']), ('feedback', bool(c['memory'])),
                           ('max_updates', 10), ('keep', 4), ('score_mode', 'stable_log'), ('tta_steps', 0)]:
            assert d['args'][field] == val
        prov = d['provenance']
        assert prov['predictor'] == m['predictor_sha256'] and prov['probe'] == c['probe_sha256']
        assert prov['code'] == m['code_sha256']['closed_loop_stable_scores.py']
        assert prov['common'] == m['code_sha256']['common.py'] and prov['meta'] == m['meta_sha256']
        assert digest(c['probe']) == c['probe_sha256']
        if c['memory']: assert prov['memory'] == c['memory_sha256'] == digest(c['memory'])
        verified = []
        for row in d['per_task']:
            stem = row['task'].replace(':', '_'); folder = Path(str(path) + '.tasks')
            tp = folder / (stem + '.json'); t = json.loads(tp.read_text()); track(tp)
            assert t['result'] == row and t['provenance'] == prov and t['args'] == d['args']
            sp = folder / t['score_file']; track(sp)
            with np.load(sp) as z:
                idx, y, scores = z['indices'], z['labels'], z['scores'][:, 1]
                np.testing.assert_array_equal(idx, np.flatnonzero(tasks == row['task']))
                np.testing.assert_array_equal(y, labels[idx]); assert len(y) == 1024 and np.isfinite(scores).all()
                assert y.sum() == row['n_solutions_in_shortlist'] == old[row['task']]['n_solutions_in_shortlist']
                with np.load(Path(m['probe_evaluation']) / stem / (c['head'] + '__base.npz')) as base:
                    np.testing.assert_array_equal(scores[0], base['log_score'])
                chosen = set(); ctx = []; updates = 0; first = 0
                for k in range(1, 101):
                    values = scores[updates].copy()
                    if chosen: values[list(chosen)] = -np.inf
                    j = int(values.argmax()); chosen.add(j); record = t['attempts'][k - 1]
                    assert record['attempt'] == k and record['candidate_local'] == j and record['pack_index'] == int(idx[j])
                    assert record['context'] == ctx == t['contexts'][updates] and record['score_snapshot'] == updates
                    assert record['score'] == scores[updates, j] and record['solved'] == bool(y[j])
                    np.testing.assert_array_equal(record['action'], actions[idx[j]].tolist()); attempts_checked += 1
                    if y[j]: first = k; break
                    if c['memory'] and updates < 10:
                        ctx = (ctx + [int(idx[j])])[-4:]; updates += 1
                        assert not labels[ctx].any()
                assert len(t['attempts']) == (first or 100)
                assert first == row['first_success'] and updates == row['updates']
                assert len(scores) == updates + 1 == len(t['contexts']) and t['contexts'][-1] == ctx
                # Telescoped log-weight sum, independent of evaluator's explicit weight array.
                auccess = float(1 - np.log(first) / np.log(101)) if first else 0.
                assert abs(auccess - row['auccess']) < 1e-12
                if c['name'] == 'base_original': assert first == old[row['task']]['first_success']
                verified.append({'task': row['task'], 'first_success': first, 'auccess': auccess})
        def agg(rows):
            f = np.array([r['first_success'] for r in rows])
            return {'n_tasks': len(rows), 'auccess': float(np.mean([r['auccess'] for r in rows])),
                'success_at_1': float(np.mean(f == 1)), 'success_at_10': float(np.mean((f > 0) & (f <= 10))),
                'success_at_100': float(np.mean(f > 0))}
        metrics = agg(verified); assert round(metrics['auccess'], 4) == d['aggregate']['auccess']
        for k in [1, 10, 100]: assert round(metrics[f'success_at_{k}'], 3) == d['aggregate']['success_at'][str(k)]
        completed[c['name']] = {'metrics': metrics, 'by_template': {temp: agg([r for r in verified if r['task'].startswith(temp)]) for temp in sorted({r['task'][:5] for r in verified})}, 'per_task': verified}
    assert completed
    result = {'complete_experiment': len(completed) == len(m['conditions']), 'completed_conditions_audit_passed': True,
        'manifest_sha256': digest(a.root / 'manifest.json'), 'auditor_sha256': digest(__file__),
        'n_completed_conditions': len(completed), 'n_registered_conditions': len(m['conditions']),
        'attempts_replayed': attempts_checked, 'conditions': completed, 'file_hashes': files,
        'scope': 'Completed configurations only; incomplete memory grid must not be summarized as a final method comparison.'}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['conditions', 'file_hashes']}), flush=True)


if __name__ == '__main__': main()
