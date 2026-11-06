"""Post-hoc loss attribution from complete cached closed-loop score traces.

Freeze/switch policies replay exactly from an observed common failure prefix;
they do not need scores for unobserved memory contexts. Thresholds are exploratory.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality(first): return float(1 - np.log(first) / np.log(101)) if first else 0.


def rank(y, scores, excluded):
    mask = np.ones(len(y), bool); mask[excluded] = False
    ordered = y[mask][np.argsort(-scores[mask], kind='stable')]
    return int(np.flatnonzero(ordered)[0]) + 1 if ordered.any() else None


def metric(rows, policy):
    first = np.array([r['policies'][policy] for r in rows])
    return {'n': len(rows), 'auccess': float(np.mean([quality(f) for f in first])),
            'success_at_10': float(np.mean((first > 0) & (first <= 10))),
            'success_at_100': float(np.mean(first > 0))}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); root = a.root; s = json.loads((root / 'SUMMARY.json').read_text())
    assert s['complete'] and s['trace_audit_passed']
    all_rows = []; conditions = {}; input_hashes = {}; cutoffs = [1, 2, 4, 10]
    for name, c in s['conditions'].items():
        if not c['memory']: continue
        base = {r['task']: r for r in s['conditions'][f"base_p{c['probe_seed']}"]['per_task']}
        rows = []
        for row in c['per_task']:
            task = row['task']; stem = task.replace(':', '_')
            folder = root / 'evaluation' / (name + '.json.tasks')
            tp = folder / (stem + '.json'); tr = json.loads(tp.read_text()); sp = folder / tr['score_file']
            input_hashes[str(tp.relative_to(root))] = sha(tp); input_hashes[str(sp.relative_to(root))] = sha(sp)
            first = row['first_success']; bf = base[task]['first_success']
            category = ('both_solved' if first else 'lost') if bf else ('gained' if first else 'both_failed')
            with np.load(sp) as z:
                y = z['labels']; scores = z['scores'][:, 1]
                choices = [t['candidate_local'] for t in tr['attempts']]
                policies = {'base': bf, 'actual': first}
                for k in cutoffs:
                    for kind in ['freeze', 'base_fallback']:
                        if first and first <= k:
                            value = first
                        else:
                            assert len(scores) > k and not y[choices[:k]].any()
                            rr = rank(y, scores[k] if kind == 'freeze' else scores[0], choices[:k])
                            candidate = k + rr if rr is not None else 101
                            value = candidate if candidate <= 100 else 0
                        policies[f'{kind}_{k}'] = value
                assert policies['freeze_10'] == first
                assert abs(quality(first) - row['auccess']) < 1e-12
                snapshots = []
                for u, values in enumerate(scores):
                    excluded = choices[:u]
                    assert not y[excluded].any()
                    rr = rank(y, values, excluded)
                    br = rank(y, scores[0], excluded)
                    snapshots.append({'update': u, 'k': len(tr['contexts'][u]),
                        'best_success_remaining_rank': rr, 'base_remaining_rank_same_prefix': br,
                        'success_attempt_if_frozen': u + rr if rr is not None else None,
                        'within_remaining_budget_if_frozen': rr is not None and u + rr <= 100})
                rec = {'condition': name, 'group': c['group'], 'memory_seed': c['memory_seed'],
                       'probe_seed': c['probe_seed'], 'task': task, 'template': task[:5],
                       'category': category, 'auccess_delta': quality(first) - quality(bf),
                       'n_solutions': int(y.sum()), 'policies': policies, 'snapshots': snapshots}
                rows.append(rec); all_rows.append(rec)
        categories = {}
        for key in ['lost', 'gained', 'both_solved', 'both_failed']:
            rr = [r for r in rows if r['category'] == key]
            categories[key] = {'n': len(rr), 'auccess_delta_contribution': sum(r['auccess_delta'] for r in rr) / 100,
                               'by_template': dict(Counter(r['template'] for r in rr))}
        assert abs(sum(v['auccess_delta_contribution'] for v in categories.values()) -
                   s['paired_comparisons'][name + '_vs_base']['delta']['auccess']) < 1e-12
        conditions[name] = {'categories': categories, 'policies': {k: metric(rows, k) for k in rows[0]['policies']},
                            'rows': rows}
    groups = {}
    for group in ['latent_only', 'latent_geometry']:
        cc = [v for k, v in conditions.items() if k.startswith(group)]
        rr = [r for r in all_rows if r['group'] == group]; lost = [r for r in rr if r['category'] == 'lost']
        common = [r for r in rr if r['category'] == 'both_solved']
        groups[group] = {
            'n_configurations': len(cc),
            'mean_categories': {k: {'n_per_100_tasks': float(np.mean([c['categories'][k]['n'] for c in cc])),
                'auccess_delta_contribution': float(np.mean([c['categories'][k]['auccess_delta_contribution'] for c in cc]))}
                for k in ['lost', 'gained', 'both_solved', 'both_failed']},
            'mean_policy_metrics': {k: {metric_name: float(np.mean([c['policies'][k][metric_name] for c in cc]))
                for metric_name in ['auccess', 'success_at_10', 'success_at_100']} for k in cc[0]['policies']},
            'lost_configuration_task_occurrences': len(lost),
            'lost_unique_tasks': sorted({r['task'] for r in lost}),
            'lost_occurrences_by_template': dict(Counter(r['template'] for r in lost)),
            'lost_base_solved_within10': sum(r['policies']['base'] <= 10 for r in lost),
            'lost_first_update_outside_budget': sum(not r['snapshots'][1]['within_remaining_budget_if_frozen'] for r in lost),
            'lost_first_update_rank_worsened': sum(r['snapshots'][1]['best_success_remaining_rank'] > r['snapshots'][1]['base_remaining_rank_same_prefix'] for r in lost),
            'lost_rescued_freeze1': sum(r['policies']['freeze_1'] > 0 for r in lost),
            'lost_rescued_base_fallback10': sum(r['policies']['base_fallback_10'] > 0 for r in lost),
            'common_solved_faster': sum(r['policies']['actual'] < r['policies']['base'] for r in common),
            'common_solved_slower': sum(r['policies']['actual'] > r['policies']['base'] for r in common),
            'common_solved_same': sum(r['policies']['actual'] == r['policies']['base'] for r in common),
            'warning': 'Occurrences overlap tasks across nine shared-model configurations, not independent samples.'}
    result = {'complete': True, 'summary_sha256': sha(root / 'SUMMARY.json'), 'script_sha256': sha(__file__),
              'scope': 'Post-hoc trace-only attribution. Frozen-context/base-fallback counterfactuals replay from shared real failure prefixes; no unobserved-context model predictions or new training. Cutoffs chosen after seeing overall result; not confirmatory.',
              'groups': groups, 'conditions': conditions, 'input_hashes': input_hashes}
    out = root / 'LOSS_ATTRIBUTION.json'; out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__': main()
