"""Zero-residual intervention on the fresh uniform128 checkpoint; no encoding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

P = Path(__file__).resolve().parents[3]
R = P / 'results/fresh_zero_feedback_eval_20261019'
T = P / 'results/query_balance_training_20261019'
C = P / 'results/query_balance_centered_eval_20261019'
S = P / 'results/q0_first_feedback_20261018'
F = P / 'results/residual_fusion_diagnostic_20261019'
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
    source, training = read(S / 'PLAN.json'), read(T / 'PLAN.json')
    plan = {'code_sha256': sha(__file__), 'source_plan_sha256': sha(S / 'PLAN.json'),
            'training_plan_sha256': sha(T / 'PLAN.json'), 'budget': 9,
            'tasks': {p: s['tasks'] for p, s in source['panels'].items()},
            'checkpoints': {p: read(T / 'models' / p / 'uniform' / 'checkpoint_128.json') for p in ['cross', 'within']},
            'smoke_tasks': {p: read(F / 'PLAN.json')['tasks'][p][0] for p in ['cross', 'within']},
            'motivation': 'Uniform128 centering improves cross12to15 and within56to57 S9. Check whether the same checkpoint without residual content has similar performance.',
            'intervention': 'After each failed attempt, call original update_state with all-zero float32 residual and actual attempted action. Memory and action inputs remain; no truth observation/encoder is called.',
            'controls': 'Same uniform128 frozen checkpoint, Q0 first scores/action, classifier1024pool, nineattemptbudget; compare original raw and centered completed rollouts.',
            'limits': ['Intervention on residual-trained checkpoint, NOT a separately trained zero-residual model', 'Zero residual may be OOD; worse performance alone does not prove semantically correct use of feedback', 'Same seed/folds, adaptive diagnostic after original results', 'Uses success/failure outcome for stopping and attempted-action history, so not a no-information policy']}
    assert training['slots'] == 128
    if (R / 'PLAN.json').exists():
        assert read(R / 'PLAN.json') == plan
    else:
        write(R / 'PLAN.json', plan)
    print('REGISTERED', flush=True)


def run(panel, shard, nshards, smoke=False):
    import numpy as np
    import torch
    import selector_feedback_run as src
    from fast_inference import CandidateRanker
    torch.set_num_threads(2)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    plan = read(R / 'PLAN.json')
    assert sha(__file__) == plan['code_sha256'] and sha(S / 'PLAN.json') == plan['source_plan_sha256']
    assert sha(T / 'PLAN.json') == plan['training_plan_sha256']
    if smoke:
        ck = read(S / 'models' / panel / 'residual' / 'complete.json')
        tasks, budget = [plan['smoke_tasks'][panel]], 2
    else:
        assert read(R / 'SMOKE.json')['passed']
        ck = plan['checkpoints'][panel]
        tasks, budget = plan['tasks'][panel][shard::nshards], plan['budget']
    assert sha(ck['checkpoint']) == ck['checkpoint_sha256']
    team = src.Team(panel, 'residual', src.data.checked())
    team.model.load_state_dict(torch.load(ck['checkpoint'], map_location='cpu', weights_only=False)['state_dict'])
    team.model.eval().requires_grad_(False)
    fixed = src.old.ident(team.model)
    if not smoke:
        assert src.old.reader_hash(team.model) == ck['reader_sha256']
    zero = torch.zeros(1, 32, 196, 128, device='cuda', dtype=torch.float32)
    prov = {'plan_sha256': sha(R / 'PLAN.json'), 'checkpoint_sha256': ck['checkpoint_sha256'], 'smoke': smoke}
    try:
        for task in tasks:
            dest = R / ('smoke' if smoke else 'evaluation') / panel / task.replace(':', '_')
            if (dest / 'result.json').exists():
                assert read(dest / 'result.json')['provenance'] == prov
                continue
            team.load(task)
            pool = team.pool
            actions = torch.from_numpy(pool['action']).cuda()
            ranker = CandidateRanker(team.model, team.head, team.rule,
                torch.from_numpy(pool['z0'].copy()).cuda().float(), actions, [f[None] for f in pool['features']])
            with np.load(S / 'evaluation' / panel / 'q0' / task.replace(':', '_') / 'scores.npz') as z:
                first = z['scores'][0].copy()
                np.testing.assert_array_equal(z['indices'], pool['action_idx'])
                np.testing.assert_array_equal(z['labels'], pool['status'] > 0)
            state = team.model.initial_state().detach().clone()
            tried = np.zeros(1024, bool)
            selected, history, events = [], [], []
            success = None
            began = time.time()
            with torch.no_grad():
                for turn in range(budget):
                    ranked, rank_s = src.timed(lambda: ranker.rank(state, tried))
                    values = ranked.scores
                    if turn == 0:
                        np.testing.assert_array_equal(values, first)
                    if smoke and turn == 1:
                        with np.load(F / 'cases' / panel / task.replace(':', '_') / 'update_1_scores.npz') as z:
                            np.testing.assert_array_equal(values, z['zero_residual'])
                    j = src.pools.choose_trial(values, tried, turn)
                    tried[j] = True
                    selected.append(j)
                    history.append(values)
                    event = {'turn': turn + 1, 'local': j, 'action_idx': int(pool['action_idx'][j]),
                             'solved': bool(pool['status'][j] > 0), 'rank_s': rank_s,
                             'update_s': 0., 'observation_s': 0.}
                    events.append(event)
                    if event['solved']:
                        success = turn + 1
                        break
                    if turn + 1 == budget:
                        break
                    def update():
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            return team.model.update_state(state, zero, actions[j:j + 1])
                    state, event['update_s'] = src.timed(update)
            assert src.old.ident(team.model) == fixed and team.truth.encoder is None and team.truth.worker is None
            if smoke:
                assert len(selected) == 2
            src.data.save_npz(dest / 'scores.npz', scores=np.stack(history), selected=np.array(selected),
                              indices=pool['action_idx'], labels=pool['status'] > 0)
            write(dest / 'result.json', {'complete': True, 'provenance': prov, 'panel': panel, 'task': task,
                'first_success': success, 'events': events, 'budget': budget, 'first_scores_exact': True,
                'model_unchanged': True, 'no_observation_or_encoder_calls': True,
                'first_zero_update_replay_exact': smoke, 'seconds': time.time() - began})
            print('ZERO', panel, task, success, round(time.time() - began, 1), flush=True)
    finally:
        team.truth.close()


def summarize(smoke=False):
    import numpy as np
    plan = read(R / 'PLAN.json')
    out = {'complete': True, 'plan_sha256': sha(R / 'PLAN.json'), 'verified_rollouts': 0,
           'scope': plan['limits'], 'panels': {}}
    for panel, all_tasks in plan['tasks'].items():
        tasks = [plan['smoke_tasks'][panel]] if smoke else all_tasks
        ck = read(S / 'models' / panel / 'residual' / 'complete.json') if smoke else plan['checkpoints'][panel]
        first = {}
        for task in tasks:
            folder = R / ('smoke' if smoke else 'evaluation') / panel / task.replace(':', '_')
            row = read(folder / 'result.json')
            assert row['complete'] and row['first_scores_exact'] and row['model_unchanged'] and row['no_observation_or_encoder_calls']
            assert row['provenance'] == {'plan_sha256': sha(R / 'PLAN.json'), 'checkpoint_sha256': ck['checkpoint_sha256'], 'smoke': smoke}
            with np.load(folder / 'scores.npz') as z, np.load(S / 'evaluation' / panel / 'q0' / task.replace(':', '_') / 'scores.npz') as ref:
                for key in ['indices', 'labels']:
                    np.testing.assert_array_equal(z[key], ref[key])
                np.testing.assert_array_equal(z['scores'][0], ref['scores'][0])
                selected = z['selected']
                assert len(selected) == len(set(selected)) == len(row['events']) == len(z['scores'])
                assert np.isfinite(z['scores']).all()
                for turn, j in enumerate(selected):
                    scores = z['scores'][turn].copy()
                    scores[selected[:turn]] = -np.inf
                    assert int(scores.argmax()) == j
                    event = row['events'][turn]
                    assert event['local'] == int(j) and event['action_idx'] == int(z['indices'][j]) and event['solved'] == bool(z['labels'][j])
                hits = np.flatnonzero(z['labels'][selected])
                actual = int(hits[0] + 1) if len(hits) else None
                assert row['first_success'] == actual and len(selected) == (actual or row['budget'])
                if smoke:
                    assert row['first_zero_update_replay_exact']
                    with np.load(F / 'cases' / panel / task.replace(':', '_') / 'update_1_scores.npz') as prior:
                        np.testing.assert_array_equal(z['scores'][1], prior['zero_residual'])
            first[task] = actual
            out['verified_rollouts'] += 1
        methods = {'zero_intervention': {'n': len(tasks), 'first_success': first,
            'counts': {f's{k}': sum(v is not None and v <= k for v in first.values()) for k in [1, 3, 5, 9]}}}
        paired = {}
        if not smoke:
            parent = read(T / 'SUMMARY_128.json')['panels'][panel]['methods']
            methods.update(uniform=parent['uniform'], q0=parent['original_q0'],
                           uniform_centered=read(C / 'SUMMARY.json')['panels'][panel]['methods']['uniform_centered'])
            for a, b in [('zero_intervention', 'uniform'), ('uniform_centered', 'zero_intervention')]:
                af, bf = methods[a]['first_success'], methods[b]['first_success']
                paired[a + '_vs_' + b] = {'gains': [t for t in tasks if af[t] is not None and bf[t] is None],
                                          'losses': [t for t in tasks if af[t] is None and bf[t] is not None]}
        out['panels'][panel] = {'methods': methods, 'paired_s9': paired}
    if smoke:
        out['passed'] = True
    write(R / ('SMOKE.json' if smoke else 'SUMMARY.json'), out)
    print(json.dumps({p: {m: v['counts'] for m, v in d['methods'].items()} for p, d in out['panels'].items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['register', 'run', 'summary'])
    p.add_argument('--panel')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    register() if a.mode == 'register' else summarize(a.smoke) if a.mode == 'summary' else run(a.panel, a.shard, a.nshards, a.smoke)
