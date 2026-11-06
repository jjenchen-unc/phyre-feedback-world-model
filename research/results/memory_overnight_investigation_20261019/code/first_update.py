"""Post-hoc first-update attribution with exactly common support/history."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

R = Path(__file__).resolve().parents[1]
D = R.parent
T = D / 'query_balance_training_20261019'
C = D / 'query_balance_centered_eval_20261019'
S = D / 'q0_first_feedback_20261018'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
helper = D / 'residual_mean_step_diagnostic_20261019/code/diagnose.py'
spec = importlib.util.spec_from_file_location('tested_ranking_metrics', helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
metrics = module.ranking_metrics


def differences(rows, candidate, reference):
    pairs = [(r['metrics'][candidate], r['metrics'][reference]) for r in rows]
    auc = [a['pair_auc'] - b['pair_auc'] for a, b in pairs if a['pair_auc'] is not None and b['pair_auc'] is not None]
    ranks = [b['best_success_rank'] - a['best_success_rank'] for a, b in pairs if a['best_success_rank'] is not None and b['best_success_rank'] is not None]
    return {'n': len(rows), 'next_success_gain': sum(a['next_success'] and not b['next_success'] for a, b in pairs),
            'next_success_loss': sum(b['next_success'] and not a['next_success'] for a, b in pairs),
            'auc_defined': len(auc), 'auc_mean_change': mean(auc) if auc else None,
            'auc_median_change': median(auc) if auc else None,
            'auc_improved': sum(v > 0 for v in auc), 'auc_worsened': sum(v < 0 for v in auc),
            'success_rank_defined': len(ranks), 'success_rank_improved': sum(v > 0 for v in ranks),
            'success_rank_worsened': sum(v < 0 for v in ranks),
            'success_rank_median_improvement': median(ranks) if ranks else None,
            'top8_gain': sum(a['success_in_top8'] and not b['success_in_top8'] for a, b in pairs),
            'top8_loss': sum(b['success_in_top8'] and not a['success_in_top8'] for a, b in pairs)}


def run(panel):
    verified_path = R / ('FRESH_PANEL_' + panel + '.json')
    verified = read(verified_path)
    assert verified['complete_panel'] and verified['verified_new_rollouts'] == 400
    assert verified['plans']['training'] == sha(T / 'PLAN.json')
    tasks = read(T / 'PLAN.json')['panels'][panel]['tasks']
    rows = []
    excluded_first_success = 0
    methods = ['q0', 'uniform', 'balanced', 'uniform_centered', 'balanced_centered']
    for task in tasks:
        with np.load(S / 'evaluation' / panel / 'q0' / task.replace(':', '_') / 'scores.npz') as z:
            labels, indices, base = z['labels'].copy(), z['indices'].copy(), z['scores'][0].copy()
            first = int(z['selected'][0])
            if labels[first]:
                excluded_first_success += 1
                continue
            q0_next = int(z['selected'][1])
        tried = np.zeros(len(labels), bool)
        tried[first] = True
        case = {'task': task, 'first_failed_action_idx': int(indices[first]), 'metrics': {'q0': metrics(base, labels, tried)}}
        assert case['metrics']['q0']['next_local'] == q0_next
        for method in methods[1:]:
            centered = method.endswith('_centered')
            kind = method.replace('_centered', '')
            root = C / 'evaluation' if centered else T / 'evaluation_128'
            with np.load(root / panel / kind / task.replace(':', '_') / 'scores.npz') as z:
                np.testing.assert_array_equal(z['indices'], indices)
                np.testing.assert_array_equal(z['labels'], labels)
                np.testing.assert_array_equal(z['scores'][0], base)
                assert int(z['selected'][0]) == first
                case['metrics'][method] = metrics(z['scores'][1], labels, tried)
                assert case['metrics'][method]['next_local'] == int(z['selected'][1])
        rows.append(case)
    out = {'panel': panel, 'complete': True, 'source_panel_verification_sha256': sha(verified_path),
           'code_sha256': sha(Path(__file__)), 'metric_helper_sha256': sha(helper),
           'scope': 'Adaptive post-hoc analysis prompted by within balanced S3/S5 regression. Same Q0 failed first attempt and same initial memory for every compared method. No new simulation or training; not a new S9 estimate.',
           'n_first_failures': len(rows), 'excluded_first_successes': excluded_first_success,
           'methods': {}, 'paired': {}, 'cases': rows}
    for method in methods:
        out['methods'][method] = {'next_successes': sum(r['metrics'][method]['next_success'] for r in rows),
                                 'success_in_static_top8': sum(r['metrics'][method]['success_in_top8'] for r in rows)}
        if method != 'q0':
            out['paired'][method + '_vs_q0'] = differences(rows, method, 'q0')
    for a, b in [('balanced', 'uniform'), ('uniform_centered', 'uniform'), ('balanced_centered', 'balanced')]:
        out['paired'][a + '_vs_' + b] = differences(rows, a, b)
    (R / ('FIRST_UPDATE_' + panel + '.json')).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', choices=['cross', 'within'], required=True)
    run(parser.parse_args().panel)
