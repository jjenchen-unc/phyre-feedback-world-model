"""New base vs scene-classifier coverage on the complete official 100k actions.

No historical world-model shortlist participates in candidate construction.
Historical inputs are used solely as numerical reference tests and initial
scene encodings. Raw model scores and cache labels are kept separately.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np

P = Path('/path/to/phyre-unc')
R = P / 'results/candidate_coverage_100k_20261016'
OLD = P / 'results/milestone_scene_baseline_20261015'
MATCHED = P / 'results/matched_base_q0_20261012'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def prepare():
    import phyre
    assert not (R / 'PLAN.json').exists()
    source = read(OLD / 'PLAN.json')
    cache = phyre.get_default_100k_cache('ball')
    assert len(cache.action_array) == 100000
    save(R / 'inputs/actions.npz', actions=cache.action_array)
    plan = {'panels': {}, 'action_count': 100000, 'top_k': 1024,
            'metric': 'Fraction of all listed tasks with >=1 SOLVED action in the method-specific top1024 legal actions',
            'candidate_selection': 'ALL official 100k cache actions; validity/occlusion only; NO old-model shortlist',
            'tie_break': 'stable original cache index ascending',
            'labels': 'Official cached simulation statuses, inspected for success only after rankings; not freshly simulated outcomes',
            'checkpoint_choice': 'User explicitly specified cross base lambda0 / within base lambda0.03 seed0; not asserted optimal',
            'base_batch': 32, 'classifier_batch': 512,
            'comparison_limit': 'Same training pairs, different architectures/objectives/representations and training compute',
            'checks': 'Historical restricted-pool data used only for numerical identity checks, never as candidate source',
            'dependencies': {}}
    for panel, spec in source['panels'].items():
        weight = 0. if panel == 'cross' else .03
        name = f'base_g{weight:g}_s0'
        done = read(MATCHED / 'models' / panel / name / 'complete.json')
        assert done['complete'] and sha(done['checkpoint']) == done['checkpoint_sha256']
        assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
        tasks = spec['tasks']
        with np.load(OLD / 'inputs' / (panel + '_tokens.npz')) as z:
            assert z['tasks'].tolist() == tasks
            save(R / 'inputs' / (panel + '_tokens.npz'), tokens=z['tokens'], tasks=z['tasks'])
        statuses = np.stack([cache.load_simulation_states(t) for t in tasks])
        assert statuses.shape == (100, 100000)
        save(R / 'inputs' / (panel + '_labels.npz'), statuses=statuses, tasks=np.array(tasks))
        plan['panels'][panel] = {'tasks': tasks, 'weight': weight, 'seed': 0,
                                'base_name': name, 'base_checkpoint': done['checkpoint'],
                                'base_sha256': done['checkpoint_sha256'],
                                'classifier_checkpoint': spec['checkpoint'],
                                'classifier_sha256': spec['checkpoint_sha256'],
                                'training_pairs': spec['training_pairs']}
        for f in [Path(done['checkpoint']), Path(spec['checkpoint']), OLD / 'inputs' / (panel + '_tokens.npz')]:
            plan['dependencies'][str(f)] = sha(f)
    for f in sorted((R / 'inputs').glob('*.npz')):
        plan['dependencies'][str(f)] = sha(f)
    for f in sorted((R / 'code').glob('*.py')):
        plan['dependencies'][str(f)] = sha(f)
    write(R / 'PLAN.json', plan)
    print('PREPARED', flush=True)


def checked():
    plan = read(R / 'PLAN.json')
    for path, digest in plan['dependencies'].items():
        assert sha(path) == digest, path
    return plan


def setup_base(panel, plan):
    import torch
    sys.path.insert(0, str(R / 'code'))
    source = runpy.run_path(str(MATCHED / 'code/matched_base_q0.py'))
    config = source['checked']()
    model, probe, head, rule, _ = source['models'](config, panel, 'base', 0)
    del probe
    model.load_state_dict(torch.load(plan['panels'][panel]['base_checkpoint'], map_location='cpu', weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    from fast_inference import PreparedScorer
    from deploy_pool_data import EncoderProxy
    return model, PreparedScorer(head, rule, max_layouts=1), EncoderProxy()


class Renderer:
    def __init__(self):
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        self.process = subprocess.Popen([str(P / 'runtime/sunflower_phyre_python'), '-u',
                                        str(R / 'code/candidate_coverage_100k_cpu.py')],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        self.path = R / 'tmp' / f'{os.uname().nodename}_{os.getpid()}.npz'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def render(self, task, indices):
        self.process.stdin.write(json.dumps({'task': task, 'indices': list(map(int, indices)), 'out': str(self.path)}) + '\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Renderer exited')
            try:
                answer = json.loads(line)
            except json.JSONDecodeError:
                continue
            assert answer['ok']
            break
        with np.load(self.path) as z:
            data = {k: z[k] for k in z.files}
        self.path.unlink()
        return data

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)


def base_scores(model, scorer, encoder, raw, actions):
    import torch
    out = []
    for begin in range(0, len(raw['indices']), 32):
        sl = slice(begin, begin + 32)
        z0 = encoder.initial(raw['frames'][sl])
        a = torch.from_numpy(actions[raw['indices'][sl]].astype(np.float32)).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(torch.from_numpy(z0).cuda().float(), a)
        out.append(scorer(pred, [f[None] for f in raw['features'][sl]]).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float64)


def scene_model(panel, plan):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    source = ast.parse((R / 'code/scene_classifier.py').read_text())
    nodes = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'SceneClassifier']
    assert len(nodes) == 1
    env = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<SceneClassifier>', 'exec'), env)
    model = env['SceneClassifier']().cuda()
    model.load_state_dict(torch.load(plan['panels'][panel]['classifier_checkpoint'], map_location='cpu', weights_only=True))
    return model.eval().requires_grad_(False)


def scene_scores(model, token, actions):
    import torch
    values = []
    token = torch.from_numpy(token[None]).cuda()
    with torch.no_grad():
        for begin in range(0, len(actions), 512):
            a = torch.from_numpy(actions[begin:begin+512].astype(np.float32)).cuda()
            values.append(model(token.expand(len(a), -1, -1), a).cpu().numpy())
    return np.concatenate(values)


def smoke(method, panel):
    import torch
    torch.set_num_threads(2)
    plan = checked()
    actions = np.load(R / 'inputs/actions.npz')['actions']
    source = read(OLD / 'PLAN.json')['panels'][panel]
    task = plan['panels'][panel]['tasks'][0]
    with np.load(source['cases'][task]['shortlist']) as z:
        ref = {k: z[k] for k in z.files}
    start = time.time()
    if method == 'base':
        model, scorer, encoder = setup_base(panel, plan)
        renderer = Renderer()
        try:
            raw = renderer.render(task, ref['action_idx'][:32])
            np.testing.assert_array_equal(raw['indices'], ref['action_idx'][:32])
            np.testing.assert_array_equal(raw['features'], ref['features'][:32])
            latent = encoder.initial(raw['frames'])
            np.testing.assert_array_equal(latent, ref['z0'][:32])
            got = base_scores(model, scorer, encoder, raw, actions)
            path = MATCHED / 'evaluation' / panel / plan['panels'][panel]['base_name'] / task.replace(':', '_') / 'scores.npz'
            with np.load(path) as z:
                expected = z['scores'][:32]
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
            clock = time.time()
            raw = renderer.render(task, np.arange(512))
            bench = base_scores(model, scorer, encoder, raw, actions)
            assert len(bench) == len(raw['indices']) and np.isfinite(bench).all()
            elapsed = time.time() - clock
        finally:
            encoder.close()
            renderer.close()
    else:
        model = scene_model(panel, plan)
        token = np.load(R / 'inputs' / (panel + '_tokens.npz'))['tokens'][0]
        got = scene_scores(model, token, actions[ref['action_idx']])
        with np.load(OLD / 'evaluation' / panel / task.replace(':', '_') / 'scores.npz') as z:
            expected = z['scores']
        np.testing.assert_array_equal(got, expected)
        clock = time.time()
        bench = scene_scores(model, token, actions[:5120])
        elapsed = time.time() - clock
        assert len(bench) == 5120 and np.isfinite(bench).all()
    write(R / 'smoke' / f'{method}_{panel}.json', {
        'passed': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'max_reference_score_error': float(np.max(np.abs(got-expected))),
        'benchmark_actions_scored': len(bench), 'benchmark_seconds': elapsed,
        'benchmark_raw_budget': 512 if method == 'base' else 5120,
        'seconds_including_setup': time.time()-start,
        'gpu': torch.cuda.get_device_name(0)})
    print('SMOKE_PASS', method, panel, len(bench), elapsed, flush=True)


def run(method, shard, shards):
    import torch
    torch.set_num_threads(2)
    plan = checked()
    actions = np.load(R / 'inputs/actions.npz')['actions']
    assignment = [(p, i, t) for p, s in plan['panels'].items() for i, t in enumerate(s['tasks'])][shard::shards]
    for panel in plan['panels']:
        guard = read(R / 'smoke' / f'{method}_{panel}.json')
        assert guard['passed'] and guard['plan_sha256'] == sha(R / 'PLAN.json')
        work = [(i,t) for p,i,t in assignment if p == panel]
        if not work:
            continue
        renderer = encoder = None
        if method == 'base':
            model, scorer, encoder = setup_base(panel, plan)
            renderer = Renderer()
        else:
            model = scene_model(panel, plan)
            tokens = np.load(R / 'inputs' / (panel + '_tokens.npz'))['tokens']
        try:
            for i, task in work:
                folder = R / 'scores' / panel / method / task.replace(':', '_')
                if (folder / 'complete.json').exists():
                    assert read(folder / 'complete.json')['plan_sha256'] == sha(R / 'PLAN.json')
                    continue
                start = time.time()
                if method == 'scene':
                    values = scene_scores(model, tokens[i], actions)
                    valid = np.ones(100000, bool)  # actual common validity applied by summary
                    assert np.isfinite(values).all()
                    save(folder / 'scores.npz', scores=values, rendered_valid=valid)
                else:
                    partial = folder / 'partial.npz'
                    if partial.exists():
                        with np.load(partial) as z:
                            values=z['scores']; valid=z['rendered_valid']; begin=int(z['next_index'])
                    else:
                        values=np.full(100000, -np.inf); valid=np.zeros(100000, bool); begin=0
                    for b in range(begin, 100000, 512):
                        raw = renderer.render(task, np.arange(b, min(b+512, 100000)))
                        scored = base_scores(model, scorer, encoder, raw, actions)
                        assert np.isfinite(scored).all()
                        values[raw['indices']] = scored
                        valid[raw['indices']] = True
                        save(partial, scores=values, rendered_valid=valid, next_index=min(b+512,100000))
                        if b % 5120 == 0:
                            print('PROGRESS', method, panel, task, b+512, time.time()-start, flush=True)
                    assert valid.sum() >= 1024
                    save(folder / 'scores.npz', scores=values, rendered_valid=valid)
                    partial.unlink()
                write(folder / 'complete.json', {'plan_sha256': sha(R / 'PLAN.json'),
                      'scores_sha256': sha(folder / 'scores.npz'), 'task': task,
                      'method': method, 'seconds_this_invocation': time.time()-start,
                      'gpu': torch.cuda.get_device_name(0)})
                print('COMPLETE', method, panel, task, time.time()-start, flush=True)
        finally:
            if encoder is not None: encoder.close()
            if renderer is not None: renderer.close()
            del model
            torch.cuda.empty_cache()


def summarize(wait=False):
    plan = checked()
    expected = [R / 'scores' / p / m / t.replace(':','_') / 'complete.json'
                for p,s in plan['panels'].items() for t in s['tasks'] for m in ['base','scene']]
    deadline=time.time()+24*3600
    while not all(f.exists() for f in expected):
        if not wait or time.time()>deadline:
            raise RuntimeError(f'Incomplete: {sum(f.exists() for f in expected)}/{len(expected)}')
        time.sleep(30)
    out={'complete':True,'plan_sha256':sha(R/'PLAN.json'),'panels':{}}
    for panel,spec in plan['panels'].items():
        labels=np.load(R/'inputs'/(panel+'_labels.npz'))['statuses']
        rows=[]
        for i,task in enumerate(spec['tasks']):
            data={}
            for method in ['base','scene']:
                folder=R/'scores'/panel/method/task.replace(':','_')
                receipt=read(folder/'complete.json')
                assert receipt['plan_sha256']==out['plan_sha256']
                assert receipt['scores_sha256']==sha(folder/'scores.npz')
                with np.load(folder/'scores.npz') as z: data[method]={k:z[k] for k in z.files}
            valid=data['base']['rendered_valid']; ids=np.flatnonzero(valid)
            assert len(ids)>=1024 and (labels[i,ids]!=0).all()
            row={'task':task,'valid_actions':len(ids),'full_pool_has_success':bool((labels[i,ids]>0).any())}
            for method in data:
                score=data[method]['scores']; assert np.isfinite(score[ids]).all()
                order=ids[np.argsort(-score[ids],kind='stable')]; top=order[:1024]
                hit=labels[i,top]>0
                # Independently verify a success exists iff its best full rank is <=1024.
                ranks=np.flatnonzero(labels[i,order]>0)
                assert bool(hit.any())==bool(len(ranks) and ranks[0]<1024)
                row[method]={'covered':bool(hit.any()),'success_actions':int(hit.sum()),
                             'first_success_rank':int(ranks[0]+1) if len(ranks) else None,
                             'top1024_ids':top.tolist()}
            rows.append(row)
        out['panels'][panel]={'n_tasks':len(rows),'full_pool_covered':sum(r['full_pool_has_success'] for r in rows),
            'methods':{m:{'covered_tasks':sum(r[m]['covered'] for r in rows),
                          'coverage':sum(r[m]['covered'] for r in rows)/len(rows)} for m in ['base','scene']},
            'base_only':[r['task'] for r in rows if r['base']['covered'] and not r['scene']['covered']],
            'scene_only':[r['task'] for r in rows if r['scene']['covered'] and not r['base']['covered']],
            'neither':[r['task'] for r in rows if not r['scene']['covered'] and not r['base']['covered']],
            'tasks':rows}
    write(R/'SUMMARY.json',out)
    print(json.dumps({p:{k:v for k,v in d.items() if k!='tasks'} for p,d in out['panels'].items()},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['prepare','smoke','run','summary'])
    ap.add_argument('--method',choices=['base','scene'])
    ap.add_argument('--panel',choices=['cross','within'])
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--wait',action='store_true')
    a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='smoke':smoke(a.method,a.panel)
    elif a.mode=='run':run(a.method,a.shard,a.shards)
    else:summarize(a.wait)
