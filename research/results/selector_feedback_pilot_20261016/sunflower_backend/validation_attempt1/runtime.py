"""A5000 execution adapter; A5000 predictor workers and canonical A6000 encoder services.

Frozen experiment/model/loss/sampling code remains unchanged. The only preflight
rewrite replaces a timing-only direct encoder call with the isolated encoder RPC.
"""
import argparse
import ast
import importlib.util
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

B = Path(__file__).resolve().parent
R = B.parent
sys.path.insert(0, str(R / 'code'))


class IsolatedEncoder:
    def __init__(self):
        from encoder_rpc import Client
        self.client = Client()
        self.request = self.client.request
        self.encoder = self
        self.last_rollout_s = 0.

    def initial(self, frames):
        return self.request('initial', frames)[0]

    def rollout(self, frames):
        result, metrics = self.request('rollout', frames)
        self.last_rollout_s = metrics['encode_s']
        self.last_rpc_metrics = metrics
        return result

    def scene_measure(self, frames):
        return self.request('scene_measure', frames)[0]

    def close(self):
        pass


def install():
    import torch
    import selector_feedback_run as run
    torch.set_num_threads(2)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert 'DISABLE_ADDMM_CUDA_LT' not in os.environ
    run.LocalEncoder = IsolatedEncoder
    run.data.EncoderProxy = IsolatedEncoder
    # This block measures input encoding latency only; no training/eval rewrite.
    tree = ast.parse((R / 'code/selector_feedback_run.py').read_text())
    preflight = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
    target = [n for n in ast.walk(preflight) if isinstance(n, ast.FunctionDef) and n.name == 'initial_scene_encode']
    assert len(target) == 1
    target[0].body = ast.parse('return enc.scene_measure(frame)').body
    module = ast.fix_missing_locations(ast.Module(body=[preflight], type_ignores=[]))
    exec(compile(module, str(B / 'runtime.py'), 'exec'), run.__dict__)
    original = run.provenance
    def provenance():
        return {**original(), 'execution_backend': run.data.sha(B / 'runtime.py'),
                'encoder_rpc': run.data.sha(B / 'encoder_rpc.py'),
                'encoding_policy': run.data.sha(B / 'ENCODING_POLICY.json'),
                'hardware': 'A5000 predictor plus shared canonical A6000 encoders',
                'cublas_workspace': ':4096:8'}
    run.provenance = provenance
    return run


def hardware_check(run, panel):
    """Check fixed reader/probe forwards against stored A6000 numerical fixtures."""
    import numpy as np
    import torch
    plan = run.data.checked()
    spec = plan['panels'][panel]
    task = spec['tasks'][0]
    source = run.data.read(run.data.P / 'results/milestone_scene_baseline_20261015/PLAN.json')
    with np.load(source['panels'][panel]['cases'][task]['shortlist']) as z:
        pool = {k: z[k] for k in z.files}
    team = run.Team(panel, 'q0', plan)
    try:
        z = torch.from_numpy(pool['z0'][:32]).cuda().float()
        a = torch.from_numpy(pool['action'][:32]).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = team.model(z, a)
        scores = team.scorer(pred, [f[None] for f in pool['features'][:32]]).cpu().numpy()
        name = 'q0_g0_s0' if panel == 'cross' else 'q0_g0.03_s0'
        reference = run.data.P / 'results/matched_base_q0_20261012/evaluation' / panel / name / task.replace(':', '_') / 'scores.npz'
        with np.load(reference) as data:
            expected = data['scores'][:32]
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(np.argsort(-scores, kind='stable'), np.argsort(-expected, kind='stable'))
        run.data.write(B / f'READER_CHECK_{panel}.json', {'passed': True, 'provenance': run.provenance(),
                       'max_abs': float(np.max(np.abs(scores - expected))), 'reference': str(reference),
                       'reference_sha256': run.data.sha(reference)})
    finally:
        team.truth.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['hardware', 'preflight', 'train', 'eval', 'summary'])
    parser.add_argument('--panel', choices=['cross', 'within'])
    parser.add_argument('--kind', choices=['residual', 'zero', 'q0', 'tta'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshards', type=int, default=1)
    args = parser.parse_args()
    run = install()
    if args.mode == 'hardware':
        hardware_check(run, args.panel)
    elif args.mode == 'summary':
        run.summary()
        result = run.data.read(R / 'SUMMARY.json')
        result['timing_limits'].append('Sunflower updater/evaluation workers use one A5000 each and share two A6000 observation encoders on Oprime. Peak memory is predictor GPU only; observation wall time includes queue/IPC. Input-latency probes also include queue/IPC.')
        run.data.write(R / 'SUMMARY.json', result)
    elif args.mode == 'eval':
        run.evaluate(args.panel, args.kind, args.shard, args.nshards)
    else:
        team = run.Team(args.panel, args.kind, run.data.checked())
        try:
            run.preflight(team) if args.mode == 'preflight' else run.train(team)
        finally:
            team.truth.close()
