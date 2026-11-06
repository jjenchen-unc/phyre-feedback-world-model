"""Full100k classifier shortlist; re-simulated outcomes only after selection."""
from adapt_runtime import *
import argparse
import numpy as np


def build(shard, nshards):
    import torch
    from runtime import IsolatedEncoder
    plan = checked()
    assert read(R/'SCENE_ENCODING.json')['passed']
    src = lr.source()
    cov = src.pools.coverage
    source = src.data.checked()
    assert source['panels']['cross']['selector'] == 'scene'
    model = cov.scene_model('cross', source)
    encoder, renderer, worker = IsolatedEncoder(), cov.Renderer(), src.data.CPUWorker()
    actions = np.load(R/'inputs/actions.npy')
    with np.load(R/'inputs/tokens.npz') as z:
        tasks, tokens = z['tasks'].tolist(), z['tokens'].copy()
    with np.load(R/'inputs/validity.npz') as z:
        assert z['tasks'].tolist() == tasks
        validity = z['valid'].copy()
    # Replay classifier arithmetic on the full original 100k array.
    with np.load(src.pools.C/'inputs/cross_tokens.npz') as z:
        token = z['tokens'][0]
    actual = cov.scene_scores(model, token, actions)
    first = source['panels']['cross']['tasks'][0]
    with np.load(src.pools.C/'scores/cross/scene'/first.replace(':', '_')/'scores.npz') as z:
        np.testing.assert_array_equal(actual, z['scores'])
    try:
        for i in range(shard, len(tasks), nshards):
            task = tasks[i]
            out = R/'pools'/task.replace(':', '_')
            if (out/'complete.json').exists():
                assert read(out/'complete.json')['plan'] == sha(R/'PLAN.json')
                continue
            start = time.time()
            scores = cov.scene_scores(model, tokens[i], actions)
            assert len(scores) == 100000 and np.isfinite(scores).all()
            ids = np.flatnonzero(validity[i])
            order = ids[np.argsort(-scores[ids], kind='stable')]
            raws, count, cursor = [], 0, 0
            while count < 1024 and cursor < len(order):
                proposed = order[cursor:cursor+min(256, 1024-count)]
                cursor += len(proposed)
                raw = renderer.render(task, proposed)
                if len(raw['indices']):
                    raws.append(raw)
                count += len(raw['indices'])
            assert count == 1024
            raw = {k: np.concatenate([r[k] for r in raws]) for k in ['indices', 'frames', 'features']}
            selected = raw['indices']
            z0 = encoder.initial(raw['frames'])
            out.mkdir(parents=True, exist_ok=True)
            labelpath = out/'labels.tmp.npy'
            worker.request(mode='labels', task=task, actions=actions[selected].tolist(), out=str(labelpath))
            status = np.load(labelpath)
            assert (status != 0).all()
            save_array(out/'z0.npy', z0)
            src.data.save_npz(out/'pool.npz', action_idx=selected, action=actions[selected].astype(np.float32),
                              simulation_action=actions[selected], features=raw['features'],
                              selection_score=scores[selected], status=status)
            src.data.save_npz(out/'selector_scores.npz', scores=scores, ranked_ids=order, checked_prefix=cursor)
            write(out/'complete.json', dict(complete=True, plan=sha(R/'PLAN.json'), task=task,
                                            classifier_replay_exact=True, seconds=time.time()-start,
                                            pool_sha256=sha(out/'pool.npz'), initial_sha256=sha(out/'z0.npy')))
            labelpath.unlink()
            print('POOL', task, time.time()-start, flush=True)
    finally:
        renderer.close()
        worker.close()
        encoder.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=16)
    a = p.parse_args()
    build(a.shard, a.nshards)
