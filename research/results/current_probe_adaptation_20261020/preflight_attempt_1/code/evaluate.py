"""Fixed-state readout diagnosis, followed by gated deployment evaluation."""
from adapt_runtime import *
import argparse
import numpy as np


def geometry(h, tokens, features, times):
    import torch
    losses, pixels = [], []
    for z, f in zip(tokens, features):
        kv = h.inp(torch.from_numpy(np.array(z)).cuda().float()) + h.space
        for o in np.flatnonzero((f[0, :, 13] < .5) & (f[0, :, 11] < .5)):
            q = torch.from_numpy(np.repeat(f[:1, o], len(times), axis=0)).cuda().float()
            xy, ang = h.read(kv, q)
            target = torch.from_numpy(f[times, o, :2].copy()).cuda().float()*256
            angle = torch.from_numpy(f[times, o, 2].copy()).cuda().float()*4*np.pi
            truth = torch.stack([angle.cos(), angle.sin()], -1)
            pixel = (xy-target).abs().mean()
            loss = 10*pixel/256 + ((ang-truth).abs().mean() if f[0, o, 4:8].argmax() == 1 else 0)
            losses.append(float(loss))
            pixels.append(float(pixel))
    return dict(loss=float(np.mean(losses)), xy_mae_px=float(np.mean(pixels)))


def fixed(shard, nshards):
    import torch
    from fast_inference import PreparedScorer
    plan = checked()
    heads = {arm: head(arm) for arm in ARMS}
    scorers = {arm: PreparedScorer(h, ['soft', 4, 1]) for arm, h in heads.items()}
    team = make_team()
    try:
        for task in plan['validation'][shard::nshards]:
            dest = R/'fixed'/task.replace(':', '_')/'result.json'
            if dest.exists():
                assert read(dest)['plan'] == sha(R/'PLAN.json')
                continue
            team.load(task)
            rows = []
            states = read(R/'cache'/task.replace(':', '_')/'complete.json')['states']
            for relative in states:
                folder = R/relative
                meta = read(folder/'metadata.json')
                configure(team, meta['variant'])
                state = torch.from_numpy(np.load(folder/'state.npy')).cuda()
                assert state.dtype == team.model.initial_state().dtype
                assert hashlib.sha256(team.pool['action_idx'].tobytes()).hexdigest() == meta['pool_indices_sha256']
                values = {arm: [] for arm in ARMS}
                with torch.no_grad():
                    for start in range(0, 1024, 32):
                        z, a = team.chunk(start)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            pred = team.model(z, a, state.expand(32, -1, -1))
                        fs = [f[None] for f in team.pool['features'][start:start+32]]
                        for arm in ARMS:
                            values[arm].append(scorers[arm](pred, fs).cpu().numpy())
                    values = {arm: np.concatenate(v) for arm, v in values.items()}
                    np.testing.assert_array_equal(values['old'], np.load(folder/'scores.npy'))
                    fs = np.load(folder/'features.npy')
                    row = dict(variant=meta['variant'], depth=meta['depth'], tried=meta['tried'], heads={})
                    for arm in ARMS:
                        item = metrics(values[arm], team.pool['status'] > 0, meta['tried'])
                        for source in ['predicted', 'observed']:
                            item[source] = geometry(heads[arm], np.load(folder/(source+'.npy')), fs, meta['times'])
                        row['heads'][arm] = item
                scorepath = R/'fixed'/task.replace(':', '_')/f"{meta['variant']}_{meta['depth']}.npz"
                lr.source().data.save_npz(scorepath, **values, labels=team.pool['status'] > 0,
                                          action_idx=team.pool['action_idx'], tried=np.array(meta['tried'], dtype=np.int64))
                rows.append(row)
            write(dest, dict(complete=True, task=task, plan=sha(R/'PLAN.json'),
                             old_scores_exact=True, rows=rows))
            print('FIXED', task, len(rows), flush=True)
    finally:
        lr.close(team)


def closed(shard, nshards):
    import torch
    from fast_inference import PreparedScorer
    plan = checked()
    decision = read(R/'DECISION.json')
    assert decision['complete']
    if not decision['run_closed_loop']:
        print('CLOSED_LOOP_SKIPPED: validation gate not met', flush=True)
        return
    cases = [(t, v, a, first) for t in plan['test'] for v in VARIANTS
             for a in ARMS[1:] for first in ['own', 'common']]
    team = make_team()
    old_head = team.head
    heads = {arm: head(arm) for arm in ARMS[1:]}
    try:
        for task, variant, arm, first in cases[shard::nshards]:
            dest = R/'closed'/arm/first/variant/task.replace(':', '_')
            if (dest/'result.json').exists():
                assert read(dest/'result.json')['plan'] == sha(R/'PLAN.json')
                continue
            team.load(task)
            configure(team, variant)
            fingerprint = lr.source().old.ident(team.model)
            team.head = heads[arm]
            team.scorer = PreparedScorer(team.head, team.rule)
            state = team.model.initial_state().detach().clone()
            tried, snapshots, success = [], [], None
            refpath = lr.S/'evaluation/cross/q0'/task.replace(':', '_')/'scores.npz'
            with np.load(refpath) as ref:
                np.testing.assert_array_equal(ref['indices'], team.pool['action_idx'])
                common_first = int(ref['scores'][0].argmax())
            with torch.no_grad():
                for turn in range(9):
                    scores = team.scores(state)
                    snapshots.append(scores.copy())
                    scores[tried] = -np.inf
                    j = common_first if first == 'common' and turn == 0 else int(scores.argmax())
                    assert j not in tried
                    tried.append(j)
                    if team.pool['status'][j] > 0:
                        success = turn+1
                        break
                    if variant != 'q0' and turn < 8:
                        state = update(team, state, j)
            assert lr.source().old.ident(team.model) == fingerprint
            lr.source().data.save_npz(dest/'scores.npz', scores=np.stack(snapshots),
                                      selected=np.array(tried), indices=team.pool['action_idx'],
                                      labels=team.pool['status'] > 0)
            write(dest/'result.json', dict(complete=True, task=task, variant=variant, arm=arm,
                                          first=first, common_first=common_first, first_success=success,
                                          model_unchanged=True, plan=sha(R/'PLAN.json')))
            print('CLOSED', task, variant, arm, first, success, flush=True)
    finally:
        team.head = old_head
        lr.close(team)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['fixed', 'closed'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=16)
    a = p.parse_args()
    (fixed if a.stage == 'fixed' else closed)(a.shard, a.nshards)
