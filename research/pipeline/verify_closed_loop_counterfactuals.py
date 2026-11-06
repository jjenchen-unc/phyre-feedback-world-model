"""Independently replay counterfactual tails one action at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root; path = root / 'LOSS_ATTRIBUTION.json'; d = json.loads(path.read_text())
    n = 0; selections = 0
    for name, c in d['conditions'].items():
        for row in c['rows']:
            stem = row['task'].replace(':', '_'); folder = root / 'evaluation' / (name + '.json.tasks')
            trace = json.loads((folder / (stem + '.json')).read_text())
            with np.load(folder / trace['score_file']) as z:
                y, scores = z['labels'], z['scores'][:, 1]
                for policy, expected in row['policies'].items():
                    if policy in ['base', 'actual']: continue
                    k = int(policy.rsplit('_', 1)[1]); use_base = policy.startswith('base_fallback')
                    attempted = np.zeros(len(y), bool); first = 0
                    for step in range(1, 101):
                        if step <= k:
                            j = trace['attempts'][step - 1]['candidate_local']
                        else:
                            values = scores[0 if use_base else k].copy(); values[attempted] = -np.inf
                            j = int(np.argmax(values))
                        assert not attempted[j]; attempted[j] = True; selections += 1
                        if y[j]: first = step; break
                    assert first == expected, (name, row['task'], policy, first, expected)
                    n += 1
                assert row['policies']['freeze_10'] == row['policies']['actual']
                assert row['policies']['base_fallback_1'] == row['policies']['base']
    result = {'passed': True, 'counterfactual_policy_task_replays': n, 'action_selections_replayed': selections,
              'loss_attribution_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'scope': 'Independent stepwise replay; freeze10 equals actual and fallback1 equals base in every case. Same reused D0, no fresh-test or optimal-cutoff claim.'}
    (root / 'LOSS_ATTRIBUTION_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
