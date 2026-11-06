"""Compare Q0 first-failure residuals on train/eval scenes; cache-only observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

P = Path(__file__).resolve().parents[3]
R = P / 'results/initial_residual_shift_20261019'
S = P / 'results/q0_first_feedback_20261018'
T = P / 'results/query_balance_training_20261019'
sys.path[:0] = [str(S / 'code'), str(S / 'sunflower_backend'), str(P / 'pipeline')]
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def register():
    source = read(S / 'PLAN.json')
    roots = [P / 'results' / name / 'observations' for name in [
        'selector_feedback_pilot_20261016', 'memory_overnight_policy_20261019',
        'query_balance_training_20261019', 'query_balance_centered_eval_20261019']]
    cases = []
    for panel in ['cross', 'within']:
        train = {}
        for line in (T / 'models' / panel / 'uniform' / 'episodes.jsonl').read_text().splitlines():
            for event in json.loads(line)['events']:
                if not event['solved'][0]:
                    key = event['task']
                    if key in train:
                        assert train[key] == event['selected'][0]
                    train[key] = event['selected'][0]
        evaluation = {}
        for task in source['panels'][panel]['tasks']:
            event = read(S / 'evaluation' / panel / 'residual' / task.replace(':', '_') / 'result.json')['events'][0]
            if not event['solved']:
                evaluation[task] = event['action_idx']
        assert not set(train) & set(evaluation)
        for split, tasks in [('train', train), ('evaluation', evaluation)]:
            for task, action in sorted(tasks.items()):
                key = f'{task.replace(":", "_")}_{action:06d}'
                bucket = str(int(hashlib.sha256(key.encode()).hexdigest(), 16) % 32)
                paths = [root / panel / bucket / (key + '.npz') for root in roots]
                cache = next((path for path in paths if path.exists()), None)
                assert cache is not None, (panel, task, action, 'No new simulations permitted')
                cases.append({'panel': panel, 'split': split, 'task': task, 'action_idx': action, 'cache': str(cache)})
    plan = {'code_sha256': sha(__file__), 'source_plan_sha256': sha(S / 'PLAN.json'),
            'readers': {p: read(T / 'models' / p / 'uniform' / 'complete.json')['reader_sha256'] for p in ['cross', 'within']},
            'cases': cases,
            'question': 'Are initial failed-support residual magnitudes/relative-motion errors shifted between updater training and evaluation scenes despite same Q0-first policy?',
            'metrics': ['full residual RMS (old diagnostic parity)', 'future latent L1', 'future residual RMS', 'true future-minus-initial latent RMS', 'residual RMS divided by true-motion RMS'],
            'limits': ['All distinct first-failure scenes once, not episode-weighted', 'Q0 initial state only, not subsequent memory states', 'Train/eval scene and template difficulty differ: descriptive distribution comparison, not causal proof of overfitting', 'Cross/within use different fixed Q0 readers; compare train/eval within each protocol', 'No new simulation/encoding or predictor/updater training; all observations must come from existing canonical caches']}
    if (R / 'PLAN.json').exists():
        assert read(R / 'PLAN.json') == plan
    else:
        write(R / 'PLAN.json', plan)
    print('REGISTERED', len(cases), flush=True)


def run(shard, nshards):
    import numpy as np
    import torch
    import selector_feedback_run as src
    torch.set_num_threads(2)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    plan = read(R / 'PLAN.json')
    assert sha(__file__) == plan['code_sha256'] and sha(S / 'PLAN.json') == plan['source_plan_sha256']
    ph = sha(R / 'PLAN.json')
    team = None
    current = None
    try:
        for case in plan['cases'][shard::nshards]:
            panel, split, task = case['panel'], case['split'], case['task']
            dest = R / 'cases' / panel / split / (task.replace(':', '_') + '.json')
            if dest.exists():
                assert read(dest)['plan_sha256'] == ph
                continue
            began = time.time()
            if current != panel:
                if team is not None:
                    team.truth.close()
                    del team
                    torch.cuda.empty_cache()
                team = src.Team(panel, 'residual', src.data.checked())
                team.model.eval().requires_grad_(False)
                assert src.old.reader_hash(team.model) == plan['readers'][panel]
                current = panel
            team.load(task)
            pool = team.pool
            matches = np.flatnonzero(pool['action_idx'] == case['action_idx'])
            assert len(matches) == 1
            j = int(matches[0])
            assert pool['status'][j] < 0
            with np.load(case['cache']) as z:
                assert str(z['task']) == task and int(z['status']) == int(pool['status'][j])
                assert str(z['plan']) == sha(src.data.SOURCE_R / 'PLAN.json')
                np.testing.assert_array_equal(z['action'], pool['simulation_action'][j])
                latent = z['latent'].copy()
            state = team.model.initial_state().detach().clone()
            with torch.no_grad():
                pred = team.predict(state, j)
                if split == 'evaluation':
                    start = j // 32 * 32
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        values = team.scorer(pred, [f[None] for f in pool['features'][start:start + 32]]).cpu().numpy()
                    with np.load(S / 'evaluation' / panel / 'q0' / task.replace(':', '_') / 'scores.npz') as z:
                        np.testing.assert_array_equal(values, z['scores'][0, start:start + 32])
                        assert int(z['indices'][z['selected'][0]]) == case['action_idx']
                truth = torch.from_numpy(latent[None]).cuda().float()
                residual = truth - pred[j % 32:j % 32 + 1].float()
                future, motion = residual[:, 1:], truth[:, 1:] - truth[:, :1]
                rms = float(future.square().mean().sqrt())
                motion_rms = float(motion.square().mean().sqrt())
                measurements = {'full_residual_rms': float(residual.square().mean().sqrt()),
                                'future_latent_l1': float(future.abs().mean()), 'future_residual_rms': rms,
                                'true_motion_rms': motion_rms,
                                'error_over_motion_rms': rms / motion_rms if motion_rms > 0 else None}
            parity = False
            parent = P / 'results/residual_fusion_diagnostic_20261019/cases' / panel / task.replace(':', '_') / 'result.json'
            if split == 'evaluation' and parent.exists():
                old = read(parent)['cases'][0]
                assert old['update_number'] == 1 and old['attempted_action_idx'] == case['action_idx']
                reference = old['variants']['real']['components']['raw_residual']['rms']
                np.testing.assert_allclose(measurements['full_residual_rms'], reference, rtol=1e-6, atol=1e-7)
                parity = True
            assert all(v is None or np.isfinite(v) for v in measurements.values())
            write(dest, dict(case, complete=True, plan_sha256=ph, measurements=measurements,
                evaluation_score_chunk_exact=(split == 'evaluation'), old_diagnostic_rms_parity=parity,
                cache_sha256=sha(case['cache']), candidate_positive_count=int((pool['status'] > 0).sum()),
                seconds=time.time() - began))
            print('CASE', panel, split, task, round(measurements['future_latent_l1'], 5), flush=True)
            del pred, truth, residual, future, motion
    finally:
        if team is not None:
            team.truth.close()


def summarize():
    import numpy as np
    plan = read(R / 'PLAN.json')
    rows = []
    for case in plan['cases']:
        row = read(R / 'cases' / case['panel'] / case['split'] / (case['task'].replace(':', '_') + '.json'))
        assert row['complete'] and row['plan_sha256'] == sha(R / 'PLAN.json')
        if case['split'] == 'evaluation':
            assert row['evaluation_score_chunk_exact']
        rows.append(row)
    out = {'complete': True, 'plan_sha256': sha(R / 'PLAN.json'), 'n': len(rows),
           'old_diagnostic_rms_parity_n': sum(r['old_diagnostic_rms_parity'] for r in rows),
           'limits': plan['limits'], 'panels': {}}
    for panel in ['cross', 'within']:
        d = {}
        for split in ['train', 'evaluation']:
            subset = [r for r in rows if r['panel'] == panel and r['split'] == split]
            fields = {}
            for field in subset[0]['measurements']:
                a = np.array([r['measurements'][field] for r in subset if r['measurements'][field] is not None])
                fields[field] = {'n': len(a), 'median': float(np.median(a)) if len(a) else None,
                                 'mean': float(np.mean(a)) if len(a) else None,
                                 'p25': float(np.quantile(a, .25)) if len(a) else None,
                                 'p75': float(np.quantile(a, .75)) if len(a) else None}
            d[split] = {'n': len(subset), 'templates': len({r['task'].split(':')[0] for r in subset}), 'metrics': fields}
        d['evaluation_over_train_median_ratio'] = {
            k: d['evaluation']['metrics'][k]['median'] / d['train']['metrics'][k]['median']
            if d['evaluation']['metrics'][k]['median'] is not None and d['train']['metrics'][k]['median'] not in [None, 0] else None
            for k in d['train']['metrics']}
        out['panels'][panel] = d
    assert out['old_diagnostic_rms_parity_n'] == 20
    write(R / 'SUMMARY.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['register', 'run', 'summary'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    register() if a.mode == 'register' else summarize() if a.mode == 'summary' else run(a.shard, a.nshards)
