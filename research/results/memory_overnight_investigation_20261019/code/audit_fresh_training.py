"""Verify final fresh training exposures against candidate labels and receipts."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path('/path/to/phyre-unc')
R = P / 'results/query_balance_training_20261019'
OUT = P / 'results/memory_overnight_investigation_20261019'
D = P / 'results/selector_feedback_pilot_20261016/pools'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    plan = read(R / 'PLAN.json')
    result = {'complete': True, 'plan_sha256': sha(R / 'PLAN.json'), 'panels': {}}
    for panel, spec in plan['panels'].items():
        pools = {}
        for task in spec['train_tasks']:
            with np.load(D / panel / task.replace(':', '_') / 'pool.npz') as z:
                pools[task] = dict(zip(map(int, z['action_idx']), map(int, z['status'])))
                assert len(pools[task]) == 1024
        arms = {}
        pair_draws = []
        pair_receipts = []
        pair_activity = []
        for kind in plan['arms']:
            folder = R / 'models' / panel / kind
            receipt = read(folder / 'complete.json')
            assert receipt['slots'] == plan['slots'] and receipt['plan_sha256'] == sha(R / 'PLAN.json')
            journal = folder / 'episodes.jsonl'
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            assert [r['slot'] for r in rows] == list(range(plan['slots']))
            assert sum(r['optimizer_step'] for r in rows) == receipt['optimizer_steps']
            assert all(len(r['events']) == 2 for r in rows)
            assert all(r['optimizer_step'] == any(e['used_updates'] > 0 for e in r['events']) for r in rows)
            events = [e for row in rows for e in row['events']]
            draws = []
            for e in events:
                labels = pools[e['task']]
                query, selected = e['query'], e['selected']
                assert len(query) == len(set(query)) == 8
                assert len(selected) == len(set(selected))
                assert not set(query) & set(selected)
                assert set(e['supports']) <= set(selected)
                assert all(labels[j] < 0 for j in e['supports'])
                assert e['used_updates'] == len(e['supports'])
                assert e['solved'] == [labels[j] > 0 for j in selected]
                assert e['cap'] in [1, 2, 4, 8]
                assert bool(e['supports']) == (not e['solved'][0])
                assert selected[0] == e['first_q0_argmax'] and e['first_rank_source'] == 'q0_predictor'
                positives = sum(labels[j] > 0 for j in query)
                assert positives == e['query_positive_count']
                if kind == 'balanced':
                    eligible = [v for j, v in labels.items() if j not in set(selected)]
                    if sum(v > 0 for v in eligible) >= 4 and sum(v < 0 for v in eligible) >= 4:
                        assert positives == 4
                draws.append((e['task'], e['cap'], selected[0]))
            active = [e for e in events if e['used_updates']]
            qcounts = [e['query_positive_count'] for e in active]
            times = [r['seconds'] for r in rows]
            arms[kind] = {
                'slots': len(rows), 'episodes': len(events), 'active_episodes': len(active),
                'planned_train_scenes': len(spec['train_tasks']),
                'drawn_unique_scenes': len({e['task'] for e in events}),
                'active_unique_scenes': len({e['task'] for e in active}),
                'active_unique_templates': len({e['task'].split(':')[0] for e in active}),
                'optimizer_steps': receipt['optimizer_steps'],
                'total_support_updates': sum(e['used_updates'] for e in active),
                'used_update_hist_active': dict(Counter(e['used_updates'] for e in active)),
                'query_positive_count_hist_active': dict(Counter(qcounts)),
                'query_positive_fraction_active': sum(qcounts) / (8 * len(active)),
                'active_episodes_without_positive_query': qcounts.count(0),
                'elapsed_s': receipt['elapsed_s'], 'slot_seconds_median': float(np.median(times)),
                'slot_seconds_p90': float(np.quantile(times, .9)),
                'journal_sha256': sha(journal), 'checkpoint_sha256': receipt['checkpoint_sha256'],
            }
            pair_draws.append(draws)
            pair_receipts.append(receipt)
            pair_activity.append(([e['used_updates'] > 0 for e in events], [r['optimizer_step'] for r in rows]))
        assert pair_draws[0] == pair_draws[1]
        assert pair_activity[0] == pair_activity[1]
        assert len({r['initial_sha256'] for r in pair_receipts}) == 1
        assert len({r['reader_sha256'] for r in pair_receipts}) == 1
        result['panels'][panel] = {'paired_scene_cap_first_action_exact': True,
                                  'paired_active_episode_and_optimizer_step_pattern_exact': True, 'arms': arms}
    (OUT / 'FRESH_TRAINING_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
