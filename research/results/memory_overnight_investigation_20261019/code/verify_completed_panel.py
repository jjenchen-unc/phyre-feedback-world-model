"""Verify all fresh raw/centered rollouts for a complete protocol panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_completed import compare

R = Path(__file__).resolve().parents[1]
RESULTS = R.parent
T = RESULTS / 'query_balance_training_20261019'
C = RESULTS / 'query_balance_centered_eval_20261019'
S = RESULTS / 'q0_first_feedback_20261018'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def verify(panel):
    training, raw_plan, center_plan = read(T / 'PLAN.json'), read(T / 'EVAL_PLAN_128.json'), read(C / 'PLAN.json')
    assert sha(T / 'PLAN.json') == raw_plan['training_plan_sha256'] == center_plan['training_plan_sha256']
    assert sha(T / 'code/train.py') == training['code_sha256'] == center_plan['training_code_sha256']
    assert sha(T / 'code/evaluate.py') == raw_plan['evaluation_code_sha256']
    assert sha(C / 'code/evaluate.py') == center_plan['evaluation_code_sha256']
    assert sha(RESULTS / 'memory_overnight_policy_20261019/code/policy.py') == center_plan['intervention_code_sha256']
    assert read(C / 'SMOKE.json')['passed']
    tasks = training['panels'][panel]['tasks']
    assert tasks == raw_plan['tasks'][panel] == center_plan['tasks'][panel]
    assert len(tasks) == len(set(tasks)) == 100
    receipts = {k: read(T / 'models' / panel / k / 'checkpoint_128.json') for k in training['arms']}
    assert all(x['slots'] == 128 and x['plan_sha256'] == sha(T / 'PLAN.json') for x in receipts.values())
    assert len({x['initial_sha256'] for x in receipts.values()}) == len({x['reader_sha256'] for x in receipts.values()}) == 1
    out = {'complete_panel': True, 'panel': panel, 'verified_new_rollouts': 0,
           'scope': 'Complete100scene panel; other protocol may still be running. Matched fresh128 checkpoints.',
           'plans': {'training': sha(T / 'PLAN.json'), 'raw': sha(T / 'EVAL_PLAN_128.json'), 'centered': sha(C / 'PLAN.json')},
           'methods': {}}
    for kind in training['arms']:
        for centered in [False, True]:
            method = kind + ('_centered' if centered else '')
            first = {}
            expected = {'checkpoint_sha256': receipts[kind]['checkpoint_sha256']}
            if centered:
                expected.update(plan_sha256=sha(C / 'PLAN.json'), smoke=False)
                root = C / 'evaluation'
            else:
                expected.update(evaluation_plan_sha256=sha(T / 'EVAL_PLAN_128.json'))
                root = T / 'evaluation_128'
            for task in tasks:
                folder = root / panel / kind / task.replace(':', '_')
                row = read(folder / 'result.json')
                assert row['complete'] and row['provenance'] == expected
                assert row['first_scores_exact'] and row['model_unchanged']
                with np.load(folder / 'scores.npz') as z, np.load(S / 'evaluation' / panel / 'residual' / task.replace(':', '_') / 'scores.npz') as ref:
                    for key in ['indices', 'labels']:
                        np.testing.assert_array_equal(z[key], ref[key])
                    np.testing.assert_array_equal(z['scores'][0], ref['scores'][0])
                    selected = z['selected']
                    assert not np.isnan(z['scores']).any() and not np.isposinf(z['scores']).any()
                    assert len(selected) == len(set(selected)) == len(row['events'])
                    assert len(z['scores']) == len(selected)
                    for turn, j in enumerate(selected):
                        scores = z['scores'][turn].copy()
                        scores[selected[:turn]] = -np.inf
                        assert int(scores.argmax()) == j
                        event = row['events'][turn]
                        assert event['local'] == int(j) and event['action_idx'] == int(z['indices'][j])
                        assert event['solved'] == bool(z['labels'][j])
                    hits = np.flatnonzero(z['labels'][selected])
                    actual = int(hits[0] + 1) if len(hits) else None
                    assert row['first_success'] == actual and len(selected) == (actual or 9)
                first[task] = actual
                out['verified_new_rollouts'] += 1
            out['methods'][method] = {'n': 100, 'first_success': first,
                'counts': {f's{k}': sum(v is not None and v <= k for v in first.values()) for k in [1, 3, 5, 9]}}
    first = {t: read(S / 'evaluation' / panel / 'q0' / t.replace(':', '_') / 'result.json')['first_success'] for t in tasks}
    out['methods']['q0'] = {'n': 100, 'first_success': first,
        'counts': {f's{k}': sum(v is not None and v <= k for v in first.values()) for k in [1, 3, 5, 9]}}
    out['paired'] = {}
    for a, b in [('balanced', 'uniform'), ('uniform_centered', 'uniform'),
                 ('balanced_centered', 'balanced'), ('balanced_centered', 'uniform_centered')]:
        out['paired'][a + '_vs_' + b] = compare(out['methods'][a]['first_success'], out['methods'][b]['first_success'])
    assert out['verified_new_rollouts'] == 400
    (R / ('FRESH_PANEL_' + panel + '.json')).write_text(json.dumps(out, indent=2) + '\n')
    print(panel, 'independently verified', out['verified_new_rollouts'], 'new rollouts')
    for method, values in out['methods'].items():
        print(method, values['counts'])
    for pair, values in out['paired'].items():
        row = values['9']
        print(pair, 'gains', len(row['gains']), 'losses', len(row['losses']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', required=True, choices=['cross', 'within'])
    verify(parser.parse_args().panel)
