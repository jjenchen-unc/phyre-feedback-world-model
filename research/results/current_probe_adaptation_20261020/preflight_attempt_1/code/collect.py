"""Collect current online states and unbiased matched probe targets."""
from adapt_runtime import *
import argparse
import numpy as np


def cache_state(team, task, variant, depth, state, tried, scores):
    import torch
    out = state_folder(task, variant, depth)
    rng = np.random.default_rng(int(hashlib.sha256(task.encode()).hexdigest()[:8], 16))
    permutation = rng.permutation(1024).tolist()
    attempted_actions = {tuple(team.pool['action'][j]) for j in tried}
    qi = [j for j in permutation if j not in tried and tuple(team.pool['action'][j]) not in attempted_actions][:8]
    assert len(qi) == 8
    times = np.sort(rng.choice(np.arange(1, 32), 8, replace=False))
    obs = team.observations(qi)
    pred = predict(team, state, qi).cpu().numpy()[:, times]
    truth = np.stack([obs[j]['latent'][times] for j in qi]).astype(np.float32)
    features = np.stack([obs[j]['features'] for j in qi]).astype(np.float32)
    np.testing.assert_array_equal(features[:, 0], team.pool['features'][qi])
    assert np.isfinite(pred).all() and np.isfinite(truth).all()
    for name, array in [('predicted', pred), ('observed', truth), ('features', features),
                        ('state', state.float().cpu().numpy()), ('scores', scores)]:
        save_array(out/(name+'.npy'), array)
    meta = dict(task=task, variant=variant, depth=depth, tried=tried.copy(), query=qi,
                query_global=team.pool['action_idx'][qi].tolist(), times=times.tolist(),
                status=team.pool['status'][qi].tolist(),
                pool_indices_sha256=hashlib.sha256(team.pool['action_idx'].tobytes()).hexdigest())
    write(out/'metadata.json', meta)
    write(out/'complete.json', dict(complete=True, plan=sha(R/'PLAN.json'),
                                   hashes={p.name: sha(p) for p in sorted(out.iterdir()) if p.suffix in ['.npy', '.json'] and p.name != 'complete.json'}))
    print('STATE', task, variant, depth, flush=True)


def collect(shard, nshards, smoke=False):
    import torch
    plan = checked()
    team = make_team()
    tasks = [plan['fit'][0]] if smoke else (plan['fit']+plan['validation'])[shard::nshards]
    try:
        for task in tasks:
            dest = R/'cache'/task.replace(':', '_')
            if (dest/'complete.json').exists():
                assert read(dest/'complete.json')['plan'] == sha(R/'PLAN.json')
                continue
            team.load(task)
            states = []
            first = None
            for variant in VARIANTS:
                configure(team, variant)
                fixed = lr.source().old.ident(team.model)
                state = team.model.initial_state().detach().clone()
                tried = []
                with torch.no_grad():
                    for depth in range(6):
                        scores = team.scores(state, verify=smoke and variant == 'q0')
                        if depth == 0:
                            if first is None:
                                first = scores.copy()
                            else:
                                np.testing.assert_array_equal(first, scores)
                        if (variant == 'q0' and depth == 0) or (variant != 'q0' and depth in [1, 3, 5]):
                            cache_state(team, task, variant, depth, state, tried, scores)
                            states.append(str(state_folder(task, variant, depth).relative_to(R)))
                        if variant == 'q0' or depth == 5:
                            break
                        masked = scores.copy()
                        masked[tried] = -np.inf
                        j = int(masked.argmax())
                        if team.pool['status'][j] > 0:
                            break
                        tried.append(j)
                        state = update(team, state, j)
                assert lr.source().old.ident(team.model) == fixed
            write(dest/'complete.json', dict(complete=True, task=task, states=states,
                                            model_unchanged=True, plan=sha(R/'PLAN.json')))
            print('SCENE_COMPLETE', task, len(states), flush=True)
    finally:
        lr.close(team)


def audit():
    plan = checked()
    entries = {}
    for split in ['fit', 'validation']:
        entries[split] = []
        for task in plan[split]:
            done = read(R/'cache'/task.replace(':', '_')/'complete.json')
            assert done['complete'] and done['model_unchanged'] and done['plan'] == sha(R/'PLAN.json')
            assert done['states']
            for relative in done['states']:
                folder = R/relative
                state = read(folder/'complete.json')
                assert state['plan'] == sha(R/'PLAN.json')
                for name, digest in state['hashes'].items():
                    assert sha(folder/name) == digest
                m = read(folder/'metadata.json')
                assert m['task'] == task and not set(m['tried']) & set(m['query'])
                entries[split].append(relative)
    write(R/'DATA_AUDIT.json', dict(complete=True, plan=sha(R/'PLAN.json'), entries=entries))
    print('DATA_AUDIT', {k: len(v) for k, v in entries.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=16)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--audit', action='store_true')
    a = p.parse_args()
    audit() if a.audit else collect(a.shard, a.nshards, a.smoke)
