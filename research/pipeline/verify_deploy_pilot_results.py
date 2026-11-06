"""Independently verify saved pilot scores, pairing, and scene-level summaries."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    summary = json.loads((root / 'SUMMARY.json').read_text())
    names = ['initial_R0', 'A_s0', 'B_s0', 'A_s1', 'B_s1']
    report = {'scope': 'Post-hoc verification and descriptive sensitivity; no new rollout or training.', 'models': {}, 'sha256': {}}
    reference = None
    for name in names:
        path = root / (name + '_fixed.json')
        report['sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = json.loads(path.read_text())
        rows = {(r['task'], r['attempt'], r['condition']): r for r in raw['rows']}
        if reference is None:
            reference = rows
        assert rows.keys() == reference.keys()
        grouped = defaultdict(list)
        for key, row in rows.items():
            for field in ['actual_k', 'context', 'query']:
                assert row[field] == reference[key][field]
            queries = row['per_query']
            assert [q['index'] for q in queries] == row['query']
            y = np.array([q['solved'] for q in queries], dtype=bool)
            assert y.tolist() == [q['solved'] for q in reference[key]['per_query']]
            scores = np.array([q['score'] for q in queries])
            assert np.isfinite(scores).all()
            auc = None
            if y.any() and (~y).any():
                diffs = scores[y, None] - scores[None, ~y]
                auc = float(((diffs > 0) + .5 * (diffs == 0)).mean())
                assert abs(auc - row['rank_auc']) < 1e-12
            else:
                assert row['rank_auc'] is None
            order = np.argsort(-scores, kind='stable')
            rank = int(np.flatnonzero(y[order])[0]) + 1 if y.any() else None
            assert rank == row['first_success_rank']
            if row['condition'] in ['base', 'empty']:
                assert queries == reference[(key[0], key[1], 'base')]['per_query']
            grouped[(key[2], key[0])].append(row)
        stats = {}
        expected = summary['initial_R0'] if name == 'initial_R0' else summary['pilot'][name]
        for condition in expected:
            tasks = {}
            for (c, task), rr in grouped.items():
                if c != condition:
                    continue
                vals = [r['rank_auc'] for r in rr if r['rank_auc'] is not None]
                tasks[task] = {'auc': float(np.mean(vals)) if vals else None,
                               'l1': float(np.mean([q['latent_l1'] for r in rr for q in r['per_query']])),
                               'prefixes': len(rr),
                               'positive_queries': sum(q['solved'] for q in rr[0]['per_query']),
                               'all_tied_prefixes': sum(len(set(q['score'] for q in r['per_query'])) == 1 for r in rr)}
            auc = float(np.mean([v['auc'] for v in tasks.values() if v['auc'] is not None]))
            l1 = float(np.mean([v['l1'] for v in tasks.values()]))
            assert abs(auc - expected[condition]['rank_auc']) < 1e-12
            assert abs(l1 - expected[condition]['latent_l1']) < 1e-12
            stats[condition] = {'auc': auc, 'l1': l1, 'auc_tasks': sum(v['auc'] is not None for v in tasks.values()), 'tasks': tasks}
        deltas = {t: v['auc'] - stats['base']['tasks'][t]['auc'] for t, v in stats['policy_context']['tasks'].items() if v['auc'] is not None}
        report['models'][name] = {'rows_verified': len(rows), 'conditions': stats,
                                  'policy_minus_base': deltas,
                                  'posthoc_excluding_00012_076_delta': float(np.mean([v for t, v in deltas.items() if t != '00012:076']))}
    for seed in [0, 1]:
        assert summary['training_counts'][f'A_s{seed}'] == summary['training_counts'][f'B_s{seed}']
    report['passed'] = True
    (root / 'RESULTS_VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'models': len(names), 'rows_verified': sum(v['rows_verified'] for v in report['models'].values()), 'auc_tasks': 7}))


if __name__ == '__main__':
    main()
