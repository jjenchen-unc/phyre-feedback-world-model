"""Adaptive centering stability check on matched fresh checkpoints; no training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

P = Path(__file__).resolve().parents[3]
R = P / 'results/query_balance_centered_eval_20261019'
T = P / 'results/query_balance_training_20261019'
S = P / 'results/q0_first_feedback_20261018'
C = P / 'results/memory_overnight_policy_20261019'
sys.path.insert(0, str(T / 'code'))
import train as tr

read, write, sha = tr.read, tr.write, tr.sha


def register():
    training = read(T / 'PLAN.json')
    plan = {
        'training_plan_sha256': sha(T / 'PLAN.json'),
        'training_code_sha256': sha(T / 'code/train.py'),
        'evaluation_code_sha256': sha(__file__),
        'intervention_code_sha256': sha(C / 'code/policy.py'),
        'motivation_summary_sha256': sha(C / 'SUMMARY.json'),
        'slot': 128, 'arms': training['arms'],
        'tasks': {p: s['tasks'] for p, s in training['panels'].items()},
        'smoke_tasks': read(C / 'PLAN.json')['validation_tasks'],
        'budget': 9,
        'design': 'Center pooled residual by per-channel time/space mean; original update function, current-state residual, frozen fresh128 checkpoint. Compare against unchanged same-checkpoint evaluation.',
        'motivation': 'Original256 checkpoint centering improved cross8to10 and within57to60 S9; test whether effect persists across newly trained query-composition checkpoints.',
        'limits': ['Adaptive experiment chosen after original intervention results', 'Same tasks and same updater initialization; checkpoint stability check, NOT independent seed/fold replication', 'Evaluation-only intervention may be outside training distribution', 'No selection of a different best setting per task'],
    }
    if (R / 'PLAN.json').exists():
        assert read(R / 'PLAN.json') == plan
    else:
        write(R / 'PLAN.json', plan)
    print('REGISTERED', flush=True)


def run(panel, kind, shard, nshards, smoke=False):
    import numpy as np
    import torch
    import selector_feedback_run as src
    from fast_inference import CandidateRanker

    torch.set_num_threads(2)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    plan = read(R / 'PLAN.json')
    assert sha(__file__) == plan['evaluation_code_sha256']
    assert sha(T / 'PLAN.json') == plan['training_plan_sha256']
    assert sha(T / 'code/train.py') == plan['training_code_sha256']
    assert sha(C / 'code/policy.py') == plan['intervention_code_sha256']
    spec = importlib.util.spec_from_file_location('centering_policy', C / 'code/policy.py')
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    if smoke:
        ck = read(S / 'models' / panel / 'residual' / 'complete.json')
        tasks = [plan['smoke_tasks'][panel]]
    else:
        assert read(R / 'SMOKE.json')['passed']
        ck = read(T / 'models' / panel / kind / 'checkpoint_128.json')
        paired = [read(T / 'models' / panel / k / 'checkpoint_128.json') for k in plan['arms']]
        assert all(x['slots'] == 128 and x['plan_sha256'] == plan['training_plan_sha256'] for x in paired)
        assert len({x['initial_sha256'] for x in paired}) == len({x['reader_sha256'] for x in paired}) == 1
        tasks = plan['tasks'][panel][shard::nshards]
    assert sha(ck['checkpoint']) == ck['checkpoint_sha256']
    team = tr.make_team(panel, kind, src.data.checked())
    team.truth.parents.insert(0, team.truth.root)
    team.truth.root = R / 'observations' / panel
    team.model.load_state_dict(torch.load(ck['checkpoint'], map_location='cpu', weights_only=False)['state_dict'])
    if not smoke:
        assert src.old.reader_hash(team.model) == ck['reader_sha256']
    team.model.eval().requires_grad_(False)
    fixed = src.old.ident(team.model)
    hook = team.model.residual_in.register_forward_pre_hook(
        lambda module, args: (policy.component(args[0], 'centered'),)
    )
    prov = {'plan_sha256': sha(R / 'PLAN.json'), 'checkpoint_sha256': ck['checkpoint_sha256'], 'smoke': smoke}
    try:
        for task in tasks:
            dest = R / ('smoke' if smoke else 'evaluation') / panel / kind / task.replace(':', '_')
            if (dest / 'result.json').exists():
                assert read(dest / 'result.json')['provenance'] == prov
                continue
            team.load(task)
            pool = team.pool
            ranker = CandidateRanker(team.model, team.head, team.rule,
                torch.from_numpy(pool['z0'].copy()).cuda().float(),
                torch.from_numpy(pool['action']).cuda(), [f[None] for f in pool['features']])
            with np.load(S / 'evaluation' / panel / 'residual' / task.replace(':', '_') / 'scores.npz') as z:
                first = z['scores'][0].copy()
                np.testing.assert_array_equal(z['indices'], pool['action_idx'])
                np.testing.assert_array_equal(z['labels'], pool['status'] > 0)
            expected = None
            if smoke:
                with np.load(C / 'evaluation' / panel / 'centered' / task.replace(':', '_') / 'scores.npz') as z:
                    expected = z['scores'].copy()
            state = team.model.initial_state().detach().clone()
            tried = np.zeros(1024, bool)
            events, selected, history = [], [], []
            success = None
            began = time.time()
            with torch.no_grad():
                for turn in range(plan['budget']):
                    ranked, rank_s = src.timed(lambda: ranker.rank(state, tried))
                    values = ranked.scores
                    if turn == 0:
                        np.testing.assert_array_equal(values, first)
                    if smoke:
                        np.testing.assert_array_equal(values, expected[turn])
                    j = src.pools.choose_trial(values, tried, turn)
                    assert not tried[j]
                    tried[j] = True
                    selected.append(j)
                    history.append(values)
                    e = {'turn': turn + 1, 'local': j, 'action_idx': int(pool['action_idx'][j]),
                         'solved': bool(pool['status'][j] > 0), 'rank_s': rank_s,
                         'update_s': 0., 'observation_s': 0.}
                    events.append(e)
                    if e['solved']:
                        success = turn + 1
                        break
                    if turn == 8:
                        break
                    obs, e['observation_s'] = src.timed(lambda: team.observations([j])[j])
                    e.update(cache_hit=obs['cache_hit'], observation_encode_s=obs['observation_encode_s'])
                    state, e['update_s'] = src.timed(lambda: team.update(state, j, obs))
            assert src.old.ident(team.model) == fixed
            if smoke:
                assert len(history) == len(expected)
            src.data.save_npz(dest / 'scores.npz', scores=np.stack(history), selected=np.array(selected),
                              indices=pool['action_idx'], labels=pool['status'] > 0)
            write(dest / 'result.json', {'complete': True, 'provenance': prov, 'panel': panel,
                'kind': kind, 'task': task, 'first_success': success, 'events': events,
                'first_scores_exact': True, 'model_unchanged': True,
                'all_scores_replay_exact': smoke, 'seconds': time.time() - began})
            print('CENTERED', panel, kind, task, success, round(time.time() - began, 1), flush=True)
    finally:
        hook.remove()
        team.truth.close()


def summarize(smoke=False):
    import numpy as np
    plan = read(R / 'PLAN.json')
    ph = sha(R / 'PLAN.json')
    out = {'complete': True, 'plan_sha256': ph, 'verified_rollouts': 0, 'panels': {}, 'scope': plan['limits']}
    for panel, tasks in plan['tasks'].items():
        methods = {}
        for kind in (['uniform'] if smoke else plan['arms']):
            first = {}
            for task in ([plan['smoke_tasks'][panel]] if smoke else tasks):
                folder = R / ('smoke' if smoke else 'evaluation') / panel / kind / task.replace(':', '_')
                row = read(folder / 'result.json')
                ck = read((S / 'models' / panel / 'residual' / 'complete.json') if smoke else (T / 'models' / panel / kind / 'checkpoint_128.json'))
                assert row['complete'] and row['first_scores_exact'] and row['model_unchanged']
                assert row['provenance'] == {'plan_sha256': ph, 'checkpoint_sha256': ck['checkpoint_sha256'], 'smoke': smoke}
                with np.load(folder / 'scores.npz') as z, np.load(S / 'evaluation' / panel / 'residual' / task.replace(':', '_') / 'scores.npz') as ref:
                    for key in ['indices', 'labels']:
                        np.testing.assert_array_equal(z[key], ref[key])
                    np.testing.assert_array_equal(z['scores'][0], ref['scores'][0])
                    selected = z['selected']
                    assert len(selected) == len(set(selected))
                    for turn, j in enumerate(selected):
                        scores = z['scores'][turn].copy()
                        scores[selected[:turn]] = -np.inf
                        assert int(scores.argmax()) == j
                        e = row['events'][turn]
                        assert e['local'] == int(j) and e['action_idx'] == int(z['indices'][j]) and e['solved'] == bool(z['labels'][j])
                    hits = np.flatnonzero(z['labels'][selected])
                    actual = int(hits[0] + 1) if len(hits) else None
                    assert actual == row['first_success'] and len(selected) == (actual or 9)
                    if smoke:
                        assert row['all_scores_replay_exact']
                        with np.load(C / 'evaluation' / panel / 'centered' / task.replace(':', '_') / 'scores.npz') as reference:
                            np.testing.assert_array_equal(z['scores'], reference['scores'])
                first[task] = actual
                out['verified_rollouts'] += 1
            methods[kind + '_centered'] = {'n': len(first), 'counts': {f's{k}': sum(v is not None and v <= k for v in first.values()) for k in [1, 3, 5, 9]}, 'first_success': first}
            if not smoke:
                parent = read(T / 'SUMMARY_128.json')['panels'][panel]['methods'][kind]
                methods[kind] = parent
                methods[kind + '_centered']['paired_s9_vs_same_checkpoint'] = {
                    'gains': [t for t in tasks if first[t] is not None and parent['first_success'][t] is None],
                    'losses': [t for t in tasks if first[t] is None and parent['first_success'][t] is not None],
                }
        out['panels'][panel] = {'methods': methods}
    if smoke:
        out['passed'] = True
    write(R / ('SMOKE.json' if smoke else 'SUMMARY.json'), out)
    print(json.dumps({p: {m: x['counts'] for m, x in d['methods'].items()} for p, d in out['panels'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['register', 'run', 'summary'])
    parser.add_argument('--panel')
    parser.add_argument('--kind', default='uniform')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshards', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.mode == 'register':
        register()
    elif args.mode == 'summary':
        summarize(args.smoke)
    else:
        run(args.panel, args.kind, args.shard, args.nshards, args.smoke)
