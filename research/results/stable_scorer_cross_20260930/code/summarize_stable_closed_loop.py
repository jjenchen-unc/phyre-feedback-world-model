"""Validate action traces and summarize matched stable-scorer closed loops."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    a = p.parse_args()
    models = ['base', 'initial_R0', 'A_s0', 'B_s0', 'A_s1', 'B_s1']
    results = {}
    w = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))
    reference = json.loads((a.project / 'results/cx_clm_base.json').read_text())
    ref = {r['task']: r for r in reference['per_task']}
    old_r0 = {r['task']: r for r in json.loads((a.project / 'results/memory_redesign_20260928/R0_closed_loop.json').read_text())['per_task']}
    for name in models:
        results[name] = {}
        for mode in ['legacy', 'stable_log']:
            path = a.root / (name + '_' + mode + '.json')
            data = json.loads(path.read_text())
            rows = data['per_task']
            assert len(rows) == 100 and {r['task'] for r in rows} == ref.keys()
            for r in rows:
                assert r['n_solutions_in_shortlist'] == ref[r['task']]['n_solutions_in_shortlist']
                parts = Path(str(path) + '.tasks')
                trace = json.loads((parts / (r['task'].replace(':', '_') + '.json')).read_text())
                assert trace['result'] == r and trace['provenance'] == data['provenance']
                with np.load(parts / trace['score_file']) as scores:
                    values = scores['scores'][:, int(mode == 'stable_log')]
                    indices, labels = scores['indices'], scores['labels']
                    assert np.isfinite(values).all()
                    tried = np.zeros(len(indices), bool)
                    context, updates, first = [], 0, 0
                    for attempt, t in enumerate(trace['attempts'], 1):
                        assert t['attempt'] == attempt and t['context'] == context
                        assert t['score_snapshot'] == updates
                        assert trace['contexts'][updates] == context
                        j = int(np.argmax(np.where(tried, -np.inf, values[updates])))
                        assert j == t['candidate_local'] and int(indices[j]) == t['pack_index']
                        assert bool(labels[j]) == t['solved'] and values[updates, j] == t['score']
                        tried[j] = True
                        if labels[j]:
                            first = attempt
                            assert attempt == len(trace['attempts'])
                            break
                        if name != 'base' and updates < 10:
                            context = (context + [int(indices[j])])[-4:]
                            updates += 1
                    assert first == r['first_success'] and updates == r['updates']
                    assert len(trace['attempts']) == (first or 100)
                    assert values.shape[0] == updates + 1
                score = float((w * (np.arange(1, 101) >= first)).sum() / w.sum()) if first else 0.
                assert abs(score - r['auccess']) < 1e-12
                if mode == 'legacy' and name in ['base', 'initial_R0']:
                    previous = ref if name == 'base' else old_r0
                    assert first == previous[r['task']]['first_success'], (name, r['task'], first)
            firsts = np.array([r['first_success'] for r in rows])
            results[name][mode] = {'auccess': float(np.mean([r['auccess'] for r in rows])),
                                   'success_at_1': float(np.mean(firsts == 1)),
                                   'success_at_10': float(np.mean((firsts > 0) & (firsts <= 10))),
                                   'success_at_100': float(np.mean(firsts > 0)),
                                   'per_task': rows}
    for mode in ['legacy', 'stable_log']:
        for name in models[1:]:
            for r, b in zip(results[name][mode]['per_task'], results['base'][mode]['per_task']):
                assert r['task'] == b['task'] and (r['first_success'] == 1) == (b['first_success'] == 1)
    comparisons = {}
    for name in models:
        old = results[name]['legacy']; new = results[name]['stable_log']
        comparisons[name] = {'stable_minus_legacy_auccess': new['auccess'] - old['auccess'],
                             'stable_minus_stable_base_auccess': new['auccess'] - results['base']['stable_log']['auccess']}
    out = {'complete': True, 'trace_audit_passed': True, 'legacy_base_R0_reproduced': True,
           'scope': 'Reused cross-f1 development panel, fixed historical shortlist, final scoring intervention only; no retraining.',
           'results': results, 'comparisons': comparisons}
    (a.root / 'SUMMARY.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = ['# Stable scorer closed-loop comparison', '', 'All traces audited; original base/R0 reproduced. Reused development set, fixed historical shortlist.', '',
             '| Model | Legacy AUCCESS | Stable AUCCESS | Stable S@100 |', '|---|---:|---:|---:|']
    for name, v in results.items():
        lines.append(f"| {name} | {v['legacy']['auccess']:.4f} | {v['stable_log']['auccess']:.4f} | {v['stable_log']['success_at_100']:.0%} |")
    (a.root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'complete': True, 'trace_audit_passed': True, 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
