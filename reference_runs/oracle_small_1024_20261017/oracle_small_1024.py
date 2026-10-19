"""Registered, resumable oracle-latent diagnostic on classifier-selected pools."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

P = Path('/path/to/phyre-project')
R = P / 'results/oracle_small_1024_20261017'
F = P / 'results/selector_feedback_pilot_20261016'
C = P / 'results/candidate_coverage_100k_20261016'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(p)


def save(p, **values):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(p)


def prepare():
    assert not (R / 'PLAN.json').exists()
    source = read(F / 'PLAN.json'); deps = {}; jobs = []
    for panel, n in [('cross', 2), ('within', 1)]:
        tasks = source['panels'][panel]['tasks']
        for template in sorted({t.split(':')[0] for t in tasks}):
            candidates = [t for t in tasks if t.split(':')[0] == template]
            selected = sorted(candidates, key=lambda t: hashlib.sha256(
                ('oracle_small_20261017/' + panel + '/' + t).encode()).hexdigest())[:n]
            for task in selected:
                jobs.append({'panel': panel, 'task': task})
                folder = F / 'pools' / panel / task.replace(':', '_')
                receipt = read(folder / 'complete.json')
                assert receipt['plan'] == sha(F / 'PLAN.json')
                for name, key in [('pool.npz', 'pool_sha256'), ('z0.npy', 'initial_sha256')]:
                    path = folder / name
                    assert sha(path) == receipt[key]
                    deps[str(path)] = receipt[key]
                folder = C / 'scores' / panel / 'base' / task.replace(':', '_')
                receipt = read(folder / 'complete.json')
                assert receipt['plan_sha256'] == sha(C / 'PLAN.json')
                path = folder / 'scores.npz'
                assert sha(path) == receipt['scores_sha256']
                deps[str(path)] = receipt['scores_sha256']
    assert Counter(j['panel'] for j in jobs) == {'cross': 10, 'within': 25}
    for path in [F / 'PLAN.json', C / 'PLAN.json', Path(__file__)] + list((F / 'code').glob('*.py')):
        deps[str(path)] = sha(path)
    write(R / 'PLAN.json', {
        'jobs': jobs, 'dependencies': deps, 'actions_per_task': 1024,
        'selection': 'SHA256 ordering within each template of existing eval panel, before outcomes; cross 2/template, within 1/template',
        'pool': 'Existing initial-scene-classifier full100k legal top1024; unchanged candidate IDs and float64 simulator actions',
        'oracle': 'Canonical A6000 true-video encoder, replace future latent only, common cached initial z0; unchanged probe and rule',
        'controls': ['base', 'q0', 'selector_static', 'oracle_latent', 'oracle_geometry'],
        'primary_question': 'Separate base predictor error, fixed probe readout error, and geometry scoring rule limitations on identical candidates',
        'diagnostics': 'Per-action future latent L1, decoded surface distance and distance MAE vs simulator, predicted-object position MAE in pixels; split success/failure',
        'metrics': 'Static S1/S9/S100, AUCCESS100, pair AUC, first success rank, pool coverage; also selector-first S9',
        'interpretation': 'Oracle latent through fixed probe/rule is a diagnostic, not guaranteed success upper bound',
        'batch_size': 32, 'no_training': True, 'no_full100k_oracle': True,
    })
    print('PREPARED', len(jobs), flush=True)


def checked():
    plan = read(R / 'PLAN.json')
    for path, digest in plan['dependencies'].items():
        assert sha(path) == digest, path
    return plan


def metrics(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels, bool)
    assert np.isfinite(scores).all() and len(scores) == len(labels)
    order = np.argsort(-scores, kind='stable')
    hits = np.flatnonzero(labels[order]); rank = int(hits[0] + 1) if len(hits) else None
    pos, neg = scores[labels], scores[~labels]
    auc = float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos)*len(neg))) if len(pos) and len(neg) else None
    forced = np.concatenate(([0], order[order != 0]))
    return {'first_success_rank': rank, 'S1': int(rank is not None and rank <= 1),
            'S9': int(rank is not None and rank <= 9), 'S100': int(rank is not None and rank <= 100),
            'AUCCESS100': float(1 - np.log(max(rank, 1))/np.log(101)) if rank is not None and rank <= 100 else 0.,
            'auc': auc, 'selector_first_S9': int(labels[forced[:9]].any()), 'covered': bool(labels.any())}


def geometry_readout(probe, prediction, initial, truth_features):
    """Same FP32 per-object readout shapes and geometry arithmetic as PreparedScorer."""
    import torch
    from common import surface_dist
    from fast_inference import PreparedProbeReadout
    layout = PreparedProbeReadout(initial, prediction.shape[1], prediction.device)
    b, nt, ns, _ = prediction.shape
    frames = [np.repeat(f, nt, axis=0) for f in layout.initial]
    errors = [[] for _ in range(b)]
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        kv = probe.inp(prediction.float().reshape(b * nt, ns, -1)) + probe.space
        for k, (j, obj) in enumerate(layout.objects):
            xy, angle = probe.read(kv[j*nt:(j+1)*nt], layout.queries[k])
            xy, angle = xy.cpu().numpy(), angle.cpu().numpy()
            frames[j][:, obj, :2] = xy / 256
            frames[j][:, obj, 2] = (np.arctan2(angle[:, 1], angle[:, 0]) / 2) / (2 * np.pi) % 1
            errors[j].append(np.abs(xy[1:] - truth_features[j][1:, obj, :2] * 256).mean())
    return np.stack([surface_dist(f) for f in frames]), np.array([np.mean(v) if v else 0. for v in errors])


def summary(plan):
    panels = {}
    for panel in ['cross', 'within']:
        rows = []
        for j in plan['jobs']:
            if j['panel'] != panel:
                continue
            path = R / 'tasks' / panel / j['task'].replace(':', '_') / 'complete.json'
            if path.exists():
                row = read(path); assert row['plan'] == sha(R / 'PLAN.json'); rows.append(row)
        methods = {}
        if rows:
            for method in rows[0]['metrics']:
                methods[method] = {}
                for key in ['S1', 'S9', 'S100', 'AUCCESS100', 'selector_first_S9', 'auc', 'covered']:
                    values = [r['metrics'][method][key] for r in rows if r['metrics'][method][key] is not None]
                    methods[method][key] = float(np.mean(values)) if values else None
                methods[method]['auc_defined_tasks'] = sum(r['metrics'][method]['auc'] is not None for r in rows)
        panels[panel] = {'completed': len(rows), 'expected': sum(j['panel'] == panel for j in plan['jobs']), 'methods': methods, 'tasks': rows}
    write(R / 'SUMMARY.json', {'plan': sha(R / 'PLAN.json'), 'complete': all(p['completed'] == p['expected'] for p in panels.values()), 'panels': panels})


def worker(smoke=False):
    plan = checked()
    sys.path.insert(0, str(F / 'code'))
    import selector_feedback_run as run
    import torch
    from attribution_a_metrics import log_score
    from common import surface_dist
    torch.set_num_threads(2)
    assert 'A6000' in torch.cuda.get_device_name(0)
    assert 'DISABLE_ADDMM_CUDA_LT' not in os.environ and 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
    parent = run.data.checked(); encoder = run.old.m.Encoder(run.old.m.PCA)
    cpu = run.data.CPUWorker(); team = None; base_model = None; prov = sha(R / 'PLAN.json')
    jobs = plan['jobs']
    if smoke:
        jobs = [next(j for j in jobs if j['panel'] == p) for p in ['cross', 'within']]
    else:
        assert read(R / 'SMOKE.json')['plan'] == prov and read(R / 'SMOKE.json')['passed']
    try:
        for job in jobs:
            panel, task = job['panel'], job['task']
            folder = R / 'tasks' / panel / task.replace(':', '_'); folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'worker.lock').open('w') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if (folder / 'complete.json').exists():
                    assert read(folder / 'complete.json')['plan'] == prov
                    continue
                if team is None or team.panel != panel:
                    if team is not None:
                        team.truth.close(); del team; del base_model; torch.cuda.empty_cache()
                    team = run.Team(panel, 'q0', parent)
                    base_model, base_probe, base_head, base_rule = run.old.load_model(panel, 'base', parent)
                    assert run.old.ident(base_head) == run.old.ident(team.head) and tuple(base_rule) == tuple(team.rule)
                    del base_probe, base_head
                pool = run.data.load_pool(panel, task)
                team.load(task)
                if not (folder / 'baselines.npz').exists():
                    with np.load(C / 'scores' / panel / 'base' / task.replace(':', '_') / 'scores.npz') as z:
                        assert z['rendered_valid'][pool['action_idx']].all()
                        base = z['scores'][pool['action_idx']]
                    with torch.no_grad():
                        q0 = team.scores(team.model.initial_state(), verify=smoke)
                    save(folder / 'baselines.npz', plan=prov, base=base, q0=q0, selector_static=pool['selection_score'], indices=pool['action_idx'], labels=pool['status'] > 0)
                with np.load(folder / 'baselines.npz') as z:
                    assert str(z['plan']) == prov
                    np.testing.assert_array_equal(z['indices'], pool['action_idx'])
                def raw(i):
                    clock = time.perf_counter(); path = folder / f'raw_{i}.npz'
                    response = cpu.request(mode='rollout', task=task, action=pool['simulation_action'][i].tolist(), out=str(path))
                    assert response['status'] == int(pool['status'][i])
                    return path, time.perf_counter() - clock
                for begin in range(0, 32 if smoke else 1024, 32):
                    out = folder / 'chunks' / f'{begin:04d}.npz'
                    if out.exists():
                        with np.load(out) as z:
                            assert str(z['plan']) == prov
                            np.testing.assert_array_equal(z['indices'], pool['action_idx'][begin:begin+32])
                        continue
                    clock = time.perf_counter(); latents = []; feats = []; sim_s = []; enc_s = []
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        future = ex.submit(raw, begin)
                        for i in range(begin, begin + 32):
                            path, secs = future.result(); sim_s.append(secs)
                            if i + 1 < begin + 32:
                                future = ex.submit(raw, i + 1)
                            with np.load(path) as z:
                                np.testing.assert_array_equal(z['action'], pool['simulation_action'][i])
                                np.testing.assert_array_equal(z['features'][0], pool['features'][i])
                                if i == begin:
                                    np.testing.assert_array_equal(encoder.initial(z['frames'][:1])[0], pool['z0'][i])
                                torch.cuda.synchronize(); enc_clock = time.perf_counter()
                                latent = encoder.rollout(z['frames']).astype(np.float32)
                                torch.cuda.synchronize(); enc_s.append(time.perf_counter() - enc_clock)
                                assert latent.shape == (32, 196, 128) and np.isfinite(latent).all()
                                latent[0] = pool['z0'][i]; latents.append(latent); feats.append(z['features'][::4].copy())
                            path.unlink()
                    with torch.no_grad():
                        truth = torch.from_numpy(np.stack(latents)).cuda()
                        initial = [f[None] for f in pool['features'][begin:begin+32]]
                        values = team.scorer(truth, initial).cpu().numpy()
                        z0 = torch.from_numpy(pool['z0'][begin:begin+32].copy()).cuda().float()
                        actions = torch.from_numpy(pool['action'][begin:begin+32]).cuda()
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            pred = base_model(z0, actions)
                        base_values = team.scorer(pred, initial).cpu().numpy()
                        with np.load(folder / 'baselines.npz') as z:
                            np.testing.assert_array_equal(base_values, z['base'][begin:begin+32])
                        l1 = (pred[:, 1:].float() - truth[:, 1:]).abs().mean(dim=(1, 2, 3)).cpu().numpy()
                        base_dist, base_xy = geometry_readout(team.head, pred, initial, feats)
                        oracle_dist, oracle_xy = geometry_readout(team.head, truth, initial, feats)
                        np.testing.assert_array_equal(log_score(base_dist, team.rule), base_values)
                        np.testing.assert_array_equal(log_score(oracle_dist, team.rule), values)
                        if smoke:
                            from fast_inference import PreparedScorer
                            reference = PreparedScorer(team.head, team.rule)(truth, initial).cpu().numpy()
                            np.testing.assert_array_equal(values, reference)
                        del truth, pred, z0, actions
                    true_dist = np.stack([surface_dist(f) for f in feats])
                    geo = log_score(true_dist, team.rule)
                    assert np.isfinite(values).all() and np.isfinite(geo).all()
                    save(out, plan=prov, indices=pool['action_idx'][begin:begin+32], labels=pool['status'][begin:begin+32] > 0,
                         oracle_latent=values, oracle_geometry=geo, simulate_s=sim_s, encode_s=enc_s, wall_s=time.perf_counter()-clock,
                         base_latent_l1=l1, base_xy_mae=base_xy, oracle_xy_mae=oracle_xy,
                         base_distance=base_dist, oracle_distance=oracle_dist, true_distance=true_dist)
                    print('CHUNK', panel, task, begin+32, 'seconds', round(time.perf_counter()-clock, 2), flush=True)
                if not smoke:
                    arrays = []; wall = 0.
                    for begin in range(0, 1024, 32):
                        with np.load(folder / 'chunks' / f'{begin:04d}.npz') as z:
                            assert str(z['plan']) == prov
                            arrays.append({k: z[k].copy() for k in ['indices', 'labels', 'oracle_latent', 'oracle_geometry', 'base_latent_l1', 'base_xy_mae', 'oracle_xy_mae', 'base_distance', 'oracle_distance', 'true_distance']}); wall += float(z['wall_s'])
                    merged = {k: np.concatenate([x[k] for x in arrays]) for k in arrays[0]}
                    np.testing.assert_array_equal(merged['indices'], pool['action_idx'])
                    np.testing.assert_array_equal(merged['labels'], pool['status'] > 0)
                    with np.load(folder / 'baselines.npz') as z:
                        for k in ['base', 'q0', 'selector_static']:
                            merged[k] = z[k].copy()
                    save(folder / 'scores.npz', **merged)
                    results = {k: metrics(merged[k], merged['labels']) for k in ['base', 'q0', 'selector_static', 'oracle_latent', 'oracle_geometry']}
                    diagnostics = {}
                    for group, mask in [('all', np.ones(1024, bool)), ('success', merged['labels']), ('failure', ~merged['labels'])]:
                        fields = {k: merged[k] for k in ['base_latent_l1', 'base_xy_mae', 'oracle_xy_mae']}
                        for prefix in ['base', 'oracle']:
                            fields[prefix+'_surface_distance_mae'] = np.abs(merged[prefix+'_distance'][:,1:] - merged['true_distance'][:,1:]).mean(axis=1)
                        diagnostics[group] = {'n': int(mask.sum()), **{k: float(v[mask].mean()) if mask.any() else None for k, v in fields.items()}}
                    write(folder / 'complete.json', {'plan': prov, 'panel': panel, 'task': task, 'actions': 1024, 'scores_sha256': sha(folder / 'scores.npz'), 'oracle_wall_s': wall, 'gpu': torch.cuda.get_device_name(0), 'metrics': results, 'diagnostics': diagnostics})
                    print('COMPLETE', panel, task, results, flush=True)
                    summary(plan)
        if smoke:
            write(R / 'SMOKE.json', {'passed': True, 'plan': prov, 'tasks': jobs, 'actions_per_panel': 32, 'checks': ['fresh simulation status agrees', 'float64 action unchanged', 'initial geometry exact', 'canonical initial latent exact', 'Q0 scores reference parity', 'base/Q0 same probe and rule', 'fresh base scores exact vs full100k archived subset', 'diagnostic geometry reproduces original scores exactly', 'finite complete trajectories/scores']})
        else:
            summary(plan)
    finally:
        cpu.close()
        if team is not None:
            team.truth.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prepare', 'smoke', 'worker', 'summary'])
    mode = ap.parse_args().mode
    if mode == 'prepare': prepare()
    elif mode == 'summary': summary(read(R / 'PLAN.json'))
    else: worker(smoke=mode == 'smoke')
