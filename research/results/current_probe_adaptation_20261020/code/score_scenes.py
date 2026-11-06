"""A6000 classifier scores: A5000 deviates at ~1e-6, so retain canonical GPU."""
import os
os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
from adapt_runtime import *
import argparse
import numpy as np


def run(shard):
    import torch
    import selector_feedback_run as src
    torch.set_num_threads(2)
    assert 'A6000' in torch.cuda.get_device_name(0)
    checked()
    cov = src.pools.coverage
    source = src.data.checked()
    model = cov.scene_model('cross', source)
    actions = np.load(R/'inputs/actions.npy')
    with np.load(src.pools.C/'inputs/cross_tokens.npz') as z:
        ref_token = z['tokens'][0]
    reference = cov.scene_scores(model, ref_token, actions)
    first = source['panels']['cross']['tasks'][0]
    with np.load(src.pools.C/'scores/cross/scene'/first.replace(':', '_')/'scores.npz') as z:
        np.testing.assert_array_equal(reference, z['scores'])
    with np.load(R/'inputs/tokens.npz') as z:
        tasks, tokens = z['tasks'].tolist(), z['tokens'].copy()
    for i in range(shard, len(tasks), 2):
        out = R/'selector_scores'/tasks[i].replace(':', '_')
        if (out/'complete.json').exists():
            assert read(out/'complete.json')['plan'] == sha(R/'PLAN.json')
            continue
        start = time.time()
        scores = cov.scene_scores(model, tokens[i], actions)
        assert len(scores) == 100000 and np.isfinite(scores).all()
        save_array(out/'scores.npy', scores)
        write(out/'complete.json', dict(complete=True, task=tasks[i], plan=sha(R/'PLAN.json'),
                                        sha256=sha(out/'scores.npy'), reference_exact=True,
                                        gpu=torch.cuda.get_device_name(0), seconds=time.time()-start))
        print('CLASSIFIER_SCORE', tasks[i], time.time()-start, flush=True)
    write(R/f'SELECTOR_COMPLETE_{shard}.json', dict(complete=True, plan=sha(R/'PLAN.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, required=True)
    run(p.parse_args().shard)
