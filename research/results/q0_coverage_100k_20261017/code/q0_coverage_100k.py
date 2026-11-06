"""Independent Q0 full100k ranking, compared with the completed base/scene run."""
import argparse
import copy
from pathlib import Path
import runpy
import shutil
import time

import numpy as np
import candidate_coverage_100k as common

P = common.P
C = P / 'results/candidate_coverage_100k_20261016'
R = P / 'results/q0_coverage_100k_20261017'
common.R = R
read, write, sha, save = common.read, common.write, common.sha, common.save


def prepare():
    assert not (R / 'PLAN.json').exists()
    parent = read(C / 'PLAN.json')
    assert read(C / 'SUMMARY.json')['complete']
    for path, digest in parent['dependencies'].items():
        assert sha(path) == digest, path
    plan = copy.deepcopy(parent)
    plan.update(method='q0', comparison_source=str(C),
                q0_state='Learned initial tokens only; no supports or online updates',
                checks='Historical Q0 scores are numerical fixtures only; all100k actions independently ranked',
                dependencies=dict(parent['dependencies']))
    (R / 'inputs').mkdir(parents=True, exist_ok=True)
    for filename in ['actions.npz', 'cross_labels.npz', 'within_labels.npz']:
        shutil.copy2(C / 'inputs' / filename, R / 'inputs' / filename)
    for panel, spec in plan['panels'].items():
        name = f"q0_g{spec['weight']:g}_s{spec['seed']}"
        receipt_path = common.MATCHED / 'models' / panel / name / 'complete.json'
        done = read(receipt_path)
        assert done['complete'] and sha(done['checkpoint']) == done['checkpoint_sha256']
        spec.update(q0_name=name, q0_checkpoint=done['checkpoint'], q0_sha256=done['checkpoint_sha256'])
        plan['dependencies'][str(receipt_path)] = sha(receipt_path)
        plan['dependencies'][done['checkpoint']] = done['checkpoint_sha256']
        task = spec['tasks'][0]
        fixture = read(common.OLD / 'PLAN.json')['panels'][panel]['cases'][task]['shortlist']
        reference = common.MATCHED / 'evaluation' / panel / name / task.replace(':', '_') / 'scores.npz'
        for path in [Path(fixture), reference]:
            plan['dependencies'][str(path)] = sha(path)
    for path in [C / 'PLAN.json', C / 'SUMMARY.json', *sorted((R / 'inputs').glob('*.npz')),
                 *sorted((R / 'code').glob('*.py'))]:
        plan['dependencies'][str(path)] = sha(path)
    write(R / 'PLAN.json', plan)
    print('PREPARED Q0 full100k cross/within', flush=True)


def setup(panel, plan):
    import torch
    source = runpy.run_path(str(common.MATCHED / 'code/matched_base_q0.py'))
    config = source['checked']()
    model, probe, head, rule, _ = source['models'](config, panel, 'q0', plan['panels'][panel]['seed'])
    del probe
    ck = torch.load(plan['panels'][panel]['q0_checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval().requires_grad_(False)
    from fast_inference import PreparedScorer
    from deploy_pool_data import EncoderProxy
    return model, PreparedScorer(head, rule, max_layouts=1), EncoderProxy()


def parent_scores(panel, task, method):
    folder = C / 'scores' / panel / method / task.replace(':', '_')
    receipt = read(folder / 'complete.json')
    assert receipt['plan_sha256'] == sha(C / 'PLAN.json')
    assert receipt['scores_sha256'] == sha(folder / 'scores.npz')
    with np.load(folder / 'scores.npz') as z:
        return {k: z[k] for k in z.files}


def smoke(panel):
    import torch
    torch.set_num_threads(2)
    plan = common.checked()
    spec = plan['panels'][panel]
    task = spec['tasks'][0]
    actions = np.load(R / 'inputs/actions.npz')['actions']
    fixture = read(common.OLD / 'PLAN.json')['panels'][panel]['cases'][task]['shortlist']
    with np.load(fixture) as z:
        ref = {k: z[k] for k in z.files}
    model, scorer, encoder = setup(panel, plan)
    initial = model.initial_tokens.detach().clone()
    renderer = common.Renderer()
    try:
        raw = renderer.render(task, ref['action_idx'][:32])
        np.testing.assert_array_equal(raw['indices'], ref['action_idx'][:32])
        np.testing.assert_array_equal(raw['features'], ref['features'][:32])
        np.testing.assert_array_equal(encoder.initial(raw['frames']), ref['z0'][:32])
        got = common.base_scores(model, scorer, encoder, raw, actions)
        path = common.MATCHED / 'evaluation' / panel / spec['q0_name'] / task.replace(':', '_') / 'scores.npz'
        with np.load(path) as z:
            np.testing.assert_array_equal(z['indices'][:32], raw['indices'])
            expected = z['scores'][:32]
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(np.argsort(-got, kind='stable'), np.argsort(-expected, kind='stable'))
        start = time.perf_counter()
        raw = renderer.render(task, np.arange(512))
        bench = common.base_scores(model, scorer, encoder, raw, actions)
        elapsed = time.perf_counter() - start
        valid = parent_scores(panel, task, 'base')['rendered_valid']
        np.testing.assert_array_equal(raw['indices'], np.flatnonzero(valid[:512]))
        assert len(bench) == len(raw['indices']) and np.isfinite(bench).all()
        assert torch.equal(initial, model.initial_tokens)
    finally:
        encoder.close()
        renderer.close()
    write(R / 'smoke' / f'q0_{panel}.json', {
        'passed': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'max_reference_score_error': float(np.max(np.abs(got - expected))),
        'initial_tokens_unchanged': True, 'common_validity_exact': True,
        'benchmark_raw_budget': 512, 'benchmark_actions_scored': len(bench),
        'benchmark_seconds': elapsed, 'gpu': torch.cuda.get_device_name(0)})
    print('SMOKE_PASS q0', panel, elapsed, flush=True)


def run(shard, shards):
    import torch
    torch.set_num_threads(2)
    plan = common.checked()
    actions = np.load(R / 'inputs/actions.npz')['actions']
    work = [(p, t) for p, spec in plan['panels'].items() for t in spec['tasks']][shard::shards]
    for panel in plan['panels']:
        guard = read(R / 'smoke' / f'q0_{panel}.json')
        assert guard['passed'] and guard['plan_sha256'] == sha(R / 'PLAN.json')
        tasks = [t for p, t in work if p == panel]
        if not tasks:
            continue
        model, scorer, encoder = setup(panel, plan)
        initial = model.initial_tokens.detach().clone()
        renderer = common.Renderer()
        try:
            for task in tasks:
                folder = R / 'scores' / panel / 'q0' / task.replace(':', '_')
                if (folder / 'complete.json').exists():
                    receipt = read(folder / 'complete.json')
                    assert receipt['plan_sha256'] == sha(R / 'PLAN.json')
                    assert receipt['scores_sha256'] == sha(folder / 'scores.npz')
                    continue
                valid = parent_scores(panel, task, 'base')['rendered_valid']
                partial = folder / 'partial.npz'
                values = np.full(100000, -np.inf)
                begin, prior_seconds = 0, 0.
                if partial.exists():
                    with np.load(partial) as z:
                        assert str(z['plan_sha256']) == sha(R / 'PLAN.json')
                        values = z['scores']; begin = int(z['next_index']); prior_seconds = float(z['elapsed_s'])
                start = time.perf_counter()
                for b in range(begin, 100000, 512):
                    end = min(b + 512, 100000)
                    raw = renderer.render(task, np.arange(b, end))
                    np.testing.assert_array_equal(raw['indices'], np.flatnonzero(valid[b:end]) + b)
                    scores = common.base_scores(model, scorer, encoder, raw, actions)
                    assert len(scores) == len(raw['indices']) and np.isfinite(scores).all()
                    values[raw['indices']] = scores
                    save(partial, scores=values, next_index=end, plan_sha256=np.array(sha(R / 'PLAN.json')),
                         elapsed_s=prior_seconds + time.perf_counter() - start)
                    if b % 5120 == 0:
                        print('PROGRESS q0', panel, task, end, time.perf_counter() - start, flush=True)
                assert np.isfinite(values[valid]).all() and np.isneginf(values[~valid]).all()
                assert torch.equal(initial, model.initial_tokens)
                save(folder / 'scores.npz', scores=values, rendered_valid=valid)
                write(folder / 'complete.json', {
                    'plan_sha256': sha(R / 'PLAN.json'), 'task': task, 'method': 'q0',
                    'scores_sha256': sha(folder / 'scores.npz'), 'initial_tokens_unchanged': True,
                    'seconds_total': prior_seconds + time.perf_counter() - start,
                    'gpu': torch.cuda.get_device_name(0)})
                partial.unlink()
                print('COMPLETE q0', panel, task, flush=True)
        finally:
            encoder.close(); renderer.close()
            del model, scorer
            torch.cuda.empty_cache()


def summarize():
    plan = common.checked()
    result = {'complete': True, 'plan_sha256': sha(R / 'PLAN.json'), 'panels': {},
              'metric': plan['metric'], 'comparison_limit': plan['comparison_limit']}
    for panel, spec in plan['panels'].items():
        with np.load(R / 'inputs' / f'{panel}_labels.npz') as z:
            assert z['tasks'].tolist() == spec['tasks']
            labels = z['statuses']
        rows = []
        for i, task in enumerate(spec['tasks']):
            sources = {m: parent_scores(panel, task, m) for m in ['base', 'scene']}
            folder = R / 'scores' / panel / 'q0' / task.replace(':', '_')
            receipt = read(folder / 'complete.json')
            assert receipt['plan_sha256'] == sha(R / 'PLAN.json')
            assert receipt['scores_sha256'] == sha(folder / 'scores.npz')
            with np.load(folder / 'scores.npz') as z:
                sources['q0'] = {k: z[k] for k in z.files}
            valid = sources['base']['rendered_valid']
            np.testing.assert_array_equal(sources['q0']['rendered_valid'], valid)
            ids = np.flatnonzero(valid)
            assert len(ids) >= 1024 and (labels[i, ids] != 0).all()
            row = {'task': task, 'valid_actions': len(ids)}
            for method, source in sources.items():
                assert np.isfinite(source['scores'][ids]).all()
                order = ids[np.argsort(-source['scores'][ids], kind='stable')]
                hits = np.flatnonzero(labels[i, order] > 0)
                rank = int(hits[0] + 1) if len(hits) else None
                row[method] = {'first_success_rank': rank, 'covered': rank is not None and rank <= 1024,
                               'top1024_ids': order[:1024].tolist()}
            rows.append(row)
        methods = {m: {'n_tasks': len(rows), 'covered_tasks': sum(r[m]['covered'] for r in rows),
                      'success_at': {str(k): sum(r[m]['first_success_rank'] is not None and
                                                r[m]['first_success_rank'] <= k for r in rows) / len(rows)
                                     for k in [1, 9, 100, 1024]}}
                   for m in ['base', 'scene', 'q0']}
        parent = read(C / 'SUMMARY.json')['panels'][panel]
        for m in ['base', 'scene']:
            assert methods[m]['covered_tasks'] == parent['methods'][m]['covered_tasks']
        result['panels'][panel] = {'methods': methods, 'tasks': rows}
    write(R / 'SUMMARY.json', result)
    print({p: v['methods'] for p, v in result['panels'].items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'smoke', 'run', 'summary'])
    ap.add_argument('--panel', choices=['cross', 'within'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=40)
    args = ap.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'smoke': smoke(args.panel)
    elif args.mode == 'run': run(args.shard, args.shards)
    else: summarize()
