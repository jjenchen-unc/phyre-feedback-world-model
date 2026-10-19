"""Matched base/Q0 geometry-weight study, with immutable original artifacts.

Both architectures use identical loss, examples, time samples, optimizer/schedule
and update count. No support collection, online adaptation or fast scorer here.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path('/path/to/phyre-project')
MEM = PROJECT / 'results/small_official_20261010'
ROOT = PROJECT / 'results/matched_base_q0_20261012'
sys.path.insert(0, str(MEM / 'code'))
sys.path.insert(0, str(MEM / 'recovery_20261010'))
import recovery_runner as m
from attribution_a_metrics import auc
from prefix_policy_predictor import PrefixPredictor
from memory_geometry_followup import geometry_loss

p = m.p
WEIGHTS = [0., .03, .1, .3]


def digest_state(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def spec_name(arch, weight, seed):
    return f'{arch}_g{weight:g}_s{seed}'


def prepare(initialization):
    assert not (ROOT / 'PLAN.json').exists(), 'Never replace a registered plan'
    settings, verified = {}, {}
    for panel in ['cross', 'within']:
        parent = m.OLD / 'manifest.json' if panel == 'cross' else m.W / 'manifest.json'
        old = m.read(parent)
        train = p.f.Pack(old['stage1_pack'])
        dev = p.f.Pack(old['stage1_dev_pack'])
        # Fixed, round-robin scene coverage; same diagnostic queries for every arm.
        by = {task: sorted(ids, key=lambda i: p.f.stable(('matched-dev', panel, i)))
              for task, ids in dev.by.items()}
        dev_ids, depth = [], 0
        while len(dev_ids) < min(400, len(dev.actions)):
            for task in sorted(by):
                if depth < len(by[task]):
                    dev_ids.append(by[task][depth])
                    if len(dev_ids) == min(400, len(dev.actions)):
                        break
            depth += 1
        n = len(train.actions)
        assert n == old['stage1']['n_train']
        tasks = m.read(MEM/'selection.json')['panels'][panel]['selected']
        assert not set(tasks) & set(train.by)
        settings[panel] = {
            'train_pack': old['stage1_pack'], 'dev_pack': old['stage1_dev_pack'],
            'n_train': n, 'updates': n // 8 * 8, 'presentations': n // 8 * 8 * 8,
            'dev_ids': dev_ids, 'tasks': tasks,
            'epoch_draw_sha256': {str(seed): hashlib.sha256(b''.join(
                p.old.epoch_indices(n, seed, ep).tobytes() for ep in range(8))).hexdigest()
                for seed in range(3)}}
        for path in [parent, m.BASES[panel], m.PROBES[panel],
                     Path(old['stage1_pack']+'_meta.npz'),
                     Path(old['stage1_dev_pack']+'_meta.npz')]:
            verified[str(path)] = m.sha(path)
        for task in tasks:
            _, _, path = m.input_case(panel, task)
            verified[str(path)] = m.sha(path)
            label = path.with_name(path.stem + '_labels.npy')
            verified[str(label)] = m.sha(label)
    dependencies = list((MEM/'code').glob('*.py')) + [
        MEM/'recovery_20261010/recovery_runner.py', m.HEAD, MEM/'selection.json']
    for path in dependencies:
        verified[str(path)] = m.sha(path)
    plan = {
        'initialization': initialization, 'panels': settings, 'weights': WEIGHTS,
        'arms': ['base', 'q0'], 'seeds_by_weight': {'0': [0], '0.03': [0],
                                                 '0.1': [0, 1, 2], '0.3': [0]},
        'epochs': 8, 'batch': 8, 'lr': 3e-4, 'weight_decay': .05, 'clip': 1.,
        'scheduler': 'OneCycleLR over full fixed update budget, pct_start=.03',
        'loss': 'future latent L1 + lambda * identical Q0 geometry_loss',
        'geometry': 'frozen original panel probe, 8 fixed future times/query, '
                    'equal query then dynamic-object/time weight, XY and bar doubled angle',
        'architecture': {'n_tokens': 32, 'mem_steps': 16, 'mem_grid': 10,
                         'checkpoint_blocks': False},
        'q0_trainable': 'backbone and initial tokens; unused updater frozen',
        'base_trainable': 'all predictor parameters; no memory tokens',
        'evaluation': {'primary': 'S@9', 'budgets': [1, 3, 5, 9, 20],
                       'candidates': 1024, 'batch': 32,
                       'scorer': 'unchanged original within / predicted_mix_s0 cross',
                       'checkpoint': 'fixed final; no best-epoch selection',
                       'diagnostics': 'fixed 400 dev trajectories; distinct from ranking pools'},
        'gpu_limit': 4, 'training_runs': 24, 'code_sha256': m.sha(__file__),
        'verified_files': verified,
        'limits': ['lambda other than .1 uses one seed, exploratory weight trends',
                   'same 100 tasks across seeds are not independent replications',
                   'historical original base uses its own initialization and latent-only loss',
                   'no new support-query/updater or TTA training in this study']}
    ROOT.mkdir(parents=True, exist_ok=True)
    m.write(ROOT/'PLAN.json', plan)
    print('PREPARED', initialization, 24, flush=True)


def checked():
    plan = m.read(ROOT/'PLAN.json')
    assert plan['code_sha256'] == m.sha(__file__)
    for path, digest in plan['verified_files'].items():
        assert m.sha(path) == digest, path
    return plan


def models(plan, panel, arch, seed):
    ck = torch.load(m.BASES[panel], map_location='cpu', weights_only=False)
    args = ck['args']
    torch.manual_seed(seed + 7919)
    base = m.Predictor(ck['Din'], ck['T'], ck['S'], args['d'], args['layers'], args['heads'])
    if plan['initialization'] == 'pretrained':
        base.load_state_dict(ck['state_dict'])
    else:
        # Match the original from-scratch predictor's time/space/output init.
        torch.nn.init.trunc_normal_(base.time, std=.02)
        torch.nn.init.trunc_normal_(base.space, std=.02)
        torch.nn.init.zeros_(base.out[1].weight)
        torch.nn.init.zeros_(base.out[1].bias)
    backbone_sha = digest_state(base.state_dict())
    if arch == 'q0':
        torch.manual_seed(seed + 7919)
        model = PrefixPredictor(base, ck['Din'], stage='initial', **plan['architecture'])
    else:
        model = base.requires_grad_(True)
    model = model.cuda()
    qc = torch.load(m.PROBES[panel], map_location='cpu', weights_only=False)
    probe = m.AttnProbe(qc['D'], qc['S']).cuda()
    probe.load_state_dict(qc['state_dict']); probe.eval().requires_grad_(False)
    hc = torch.load(m.HEAD, map_location='cpu', weights_only=False) if panel == 'cross' else qc
    head = m.AttnProbe(hc['D'], hc['S']).cuda()
    head.load_state_dict(hc['state_dict']); head.eval().requires_grad_(False)
    return model, probe, head, qc['rule'], backbone_sha


def terms(model, probe, source, ids):
    z = torch.from_numpy(source.z(ids)).cuda().float()
    actions = torch.from_numpy(source.actions[ids]).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(z[:, 0].contiguous(), actions).float()
    l1 = (pred[:, 1:] - z[:, 1:]).abs().mean()
    geom = geometry_loss(probe, pred, [source.feature(i) for i in ids],
                         p.old.target_times(ids))
    return l1, geom


@torch.no_grad()
def diagnostics(model, probe, source, ids):
    model.eval(); total_l1 = total_g = 0.
    for start in range(0, len(ids), 8):
        batch = ids[start:start+8]
        l1, g = terms(model, probe, source, batch)
        total_l1 += float(l1) * len(batch)
        total_g += float(g) * len(batch)
    return {'n': len(ids), 'latent_l1': total_l1/len(ids), 'geometry': total_g/len(ids)}


def optimizer(model, total):
    params = [v for v in model.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=total,
                                                   pct_start=.03)
    return params, opt, scheduler


def train(plan, panel, arch, weight, seed):
    name = spec_name(arch, weight, seed)
    out = ROOT/'models'/panel/name
    out.mkdir(parents=True, exist_ok=True)
    prov = {'plan': m.sha(ROOT/'PLAN.json'), 'panel': panel, 'arch': arch,
            'weight': weight, 'seed': seed}
    if (out/'complete.json').exists():
        result = m.read(out/'complete.json')
        assert result['provenance'] == prov
        assert m.sha(result['checkpoint']) == result['checkpoint_sha256']
        return result
    spec = plan['panels'][panel]
    model, probe, head, rule, backbone_sha = models(plan, panel, arch, seed)
    initial_sha = digest_state(model.state_dict())
    frozen_sha = digest_state({n: v for n, v in model.named_parameters() if not v.requires_grad})
    probe_sha = digest_state(probe.state_dict())
    source, dev = p.f.Pack(spec['train_pack']), p.f.Pack(spec['dev_pack'])
    params, opt, scheduler = optimizer(model, spec['updates'])
    start = 0; history = []; elapsed_before = 0.
    resume = out/'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        assert saved['provenance'] == prov and saved['initial_sha256'] == initial_sha
        model.load_state_dict(saved['state_dict'])
        opt.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start = saved['updates']; history = saved['history']; elapsed_before = saved['seconds']
    else:
        m.write(out/'initial.json', {'provenance': prov, 'backbone_sha256': backbone_sha,
                                    'model_sha256': initial_sha,
                                    'dev': diagnostics(model, probe, dev, spec['dev_ids'])})
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter(); window = []; order = None
    for slot in range(start, spec['updates']):
        epoch, offset = divmod(slot, spec['n_train']//8)
        if order is None or offset == 0:
            order = p.old.epoch_indices(spec['n_train'], seed, epoch)
        ids = order[offset].tolist()
        model.train(); opt.zero_grad(set_to_none=True)
        l1, geom = terms(model, probe, source, ids)
        loss = l1 + weight * geom
        assert torch.isfinite(loss)
        loss.backward()
        if arch == 'q0' and slot < 2:
            assert model.initial_tokens.grad is not None
            # Zero output projection makes all backbone/token gradients zero on
            # the first scratch step; they become active after that projection learns.
            if plan['initialization'] == 'pretrained' or slot == 1:
                assert model.initial_tokens.grad.abs().sum() > 0
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        assert torch.isfinite(norm)
        opt.step(); scheduler.step()
        window.append([float(l1.detach()), float(geom.detach()), float(loss.detach())])
        done = slot + 1
        if done % 200 == 0 or done == spec['updates']:
            torch.cuda.synchronize()
            means = np.mean(window, axis=0).tolist(); window = []
            progress = {'updates': done, 'total': spec['updates'], 'epoch': epoch+1,
                        'l1': means[0], 'geometry': means[1], 'loss': means[2],
                        'weighted_geometry_over_l1': weight*means[1]/means[0],
                        'seconds': elapsed_before+time.perf_counter()-t0,
                        'peak_GiB': torch.cuda.max_memory_allocated()/2**30,
                        'lr': scheduler.get_last_lr()[0]}
            m.write(out/'progress.json', progress)
            print('TRAIN', panel, name, progress, flush=True)
        epoch_end = done % (spec['n_train']//8) == 0
        if epoch_end:
            metrics = diagnostics(model, probe, dev, spec['dev_ids'])
            history.append({'epoch': epoch+1, 'updates': done, **metrics})
            m.write(out/f'dev_epoch_{epoch+1}.json', history[-1])
        if done % 1000 == 0 or epoch_end:
            assert digest_state(probe.state_dict()) == probe_sha
            assert digest_state({n:v for n,v in model.named_parameters() if not v.requires_grad}) == frozen_sha
            atomic_save(resume, {'provenance': prov, 'initial_sha256': initial_sha,
                                'state_dict': model.state_dict(), 'optimizer': opt.state_dict(),
                                'scheduler': scheduler.state_dict(), 'updates': done,
                                'history': history, 'seconds': elapsed_before+time.perf_counter()-t0})
    checkpoint = out/'model.pt'
    atomic_save(checkpoint, {'provenance': prov, 'state_dict': model.state_dict(),
                            'updates': spec['updates'], 'architecture': plan['architecture']})
    result = {'complete': True, 'provenance': prov, 'checkpoint': str(checkpoint),
              'checkpoint_sha256': m.sha(checkpoint), 'updates': spec['updates'],
              'presentations': spec['presentations'], 'history': history,
              'initial_model_sha256': initial_sha, 'initial_backbone_sha256': backbone_sha,
              'draw_sha256': spec['epoch_draw_sha256'][str(seed)],
              'seconds': elapsed_before+time.perf_counter()-t0,
              'frozen_probe_unchanged': True, 'inactive_updater_unchanged': True}
    m.write(out/'complete.json', result)
    del model, probe, head, source, dev, opt, scheduler
    gc.collect(); torch.cuda.empty_cache()
    return result


@torch.no_grad()
def evaluate(plan, panel, arch, weight, seed):
    name = spec_name(arch, weight, seed)
    complete = m.read(ROOT/'models'/panel/name/'complete.json')
    assert m.sha(complete['checkpoint']) == complete['checkpoint_sha256']
    model, probe, head, rule, _ = models(plan, panel, arch, seed)
    saved = torch.load(complete['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(saved['state_dict']); model.eval().requires_grad_(False)
    score = p.score_function(head, rule)
    out = ROOT/'evaluation'/panel/name; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in plan['panels'][panel]['tasks']:
        path = out/task.replace(':', '_'); d, status, shortlist = m.input_case(panel, task)
        prov = {**complete['provenance'], 'checkpoint_sha256': complete['checkpoint_sha256'],
                'shortlist_sha256': m.sha(shortlist),
                'labels_sha256': m.sha(shortlist.with_name(shortlist.stem+'_labels.npy'))}
        if (path/'result.json').exists():
            row = m.read(path/'result.json'); assert row['provenance'] == prov
        else:
            z0 = torch.from_numpy(d['z0']).cuda().float()
            actions = torch.from_numpy(d['action']).cuda()
            features = [v[None] for v in d['features']]
            values = p.static_scores(model, z0, actions, features, score)
            assert np.isfinite(values).all()
            path.mkdir(parents=True, exist_ok=True)
            p.f.save_npz(path/'scores.npz', {'indices': d['action_idx'], 'labels': status>0,
                                          'scores': values})
            row = {'task': task, 'provenance': prov, 'first_success': p.first_rank(values, status>0),
                   'auc': auc(status>0, values)}
            m.write(path/'result.json', row)
        with np.load(path/'scores.npz') as z:
            np.testing.assert_array_equal(z['indices'], d['action_idx'])
            np.testing.assert_array_equal(z['labels'], status>0)
            assert p.first_rank(z['scores'], z['labels']) == row['first_success']
            assert auc(z['labels'], z['scores']) == row['auc']
        rows.append(row)
        m.write(out/'progress.json', {'n': len(rows), 'total': 100})
        print('EVAL', panel, name, len(rows), task, row['first_success'], flush=True)
    result = {'complete': True, 'raw_audit_passed': True, 'provenance': complete['provenance'],
              'n': len(rows), 'counts': {f's{k}': sum(r['first_success'] is not None and
                r['first_success']<=k for r in rows) for k in [1,3,5,9,20]},
              'auc_n': sum(r['auc'] is not None for r in rows),
              'mean_auc': float(np.mean([r['auc'] for r in rows if r['auc'] is not None])),
              'first_success': {r['task']: r['first_success'] for r in rows}}
    m.write(out/'complete.json', result)
    del model, probe, head
    gc.collect(); torch.cuda.empty_cache()


def smoke(plan, panel):
    spec = plan['panels'][panel]; source = p.f.Pack(spec['train_pack'])
    ids = p.old.epoch_indices(spec['n_train'], 0, 0)[0].tolist()
    results = {}; backbone_hash = None; initial_terms = None
    for arch in ['base', 'q0']:
        model, probe, head, rule, initial = models(plan, panel, arch, 0)
        if backbone_hash is None:
            backbone_hash = initial
        assert initial == backbone_hash
        model.train()
        l1, geom = terms(model, probe, source, ids)
        assert torch.isfinite(l1) and torch.isfinite(geom)
        if plan['initialization'] == 'scratch':
            backbone = model.base if arch == 'q0' else model
            assert torch.count_nonzero(backbone.out[1].weight) == 0
            assert torch.count_nonzero(backbone.out[1].bias) == 0
            if initial_terms is None:
                initial_terms = (float(l1.detach()), float(geom.detach()))
            assert initial_terms == (float(l1.detach()), float(geom.detach()))
        if arch == 'q0':
            reference, old_l1, old_g = p.old.q0_loss(model, probe, source, ids)
            assert torch.equal(l1, old_l1) and torch.equal(geom, old_g)
            assert torch.equal(l1+.1*geom, reference)
            del reference, old_l1, old_g
        frozen = digest_state(probe.state_dict())
        initial_weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        # Verify numerical equivalence of checkpoint removal before using it.
        outputs = []
        for checkpoint in ([True, False] if arch == 'q0' else [False]):
            if arch == 'q0': model.checkpoint_blocks = checkpoint
            model.load_state_dict(initial_weights)
            params, opt, scheduler = optimizer(model, spec['updates'])
            logs = []; torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            for slot in range(8):
                batch = p.old.epoch_indices(spec['n_train'], 0, 0)[slot].tolist()
                opt.zero_grad(set_to_none=True)
                ll, gg = terms(model, probe, source, batch); loss = ll+.1*gg
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                assert torch.isfinite(norm) and norm>0
                if arch == 'q0': assert model.initial_tokens.grad is not None
                opt.step(); scheduler.step(); logs.append([float(ll.detach()), float(gg.detach())])
            torch.cuda.synchronize()
            outputs.append({'state': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            'losses': logs, 'seconds': time.perf_counter()-start,
                            'peak_GiB': torch.cuda.max_memory_allocated()/2**30})
        if arch == 'q0':
            assert outputs[0]['losses'] == outputs[1]['losses']
            assert all(torch.equal(outputs[0]['state'][k], outputs[1]['state'][k])
                       for k in outputs[0]['state'])
        assert digest_state(probe.state_dict()) == frozen
        results[arch] = {'backbone_sha256': initial, 'geometry_same_as_q0': True,
                         'checkpoint_removal_exact': arch == 'q0',
                         'timings': [{k:v for k,v in o.items() if k!='state'} for o in outputs]}
        del model, probe, head, outputs, initial_weights, opt, scheduler, params, l1, geom
        gc.collect(); torch.cuda.empty_cache()
    m.write(ROOT/f'SMOKE_{panel}.json', {'passed': True, 'plan': m.sha(ROOT/'PLAN.json'),
                                      'models': results})
    print('SMOKE_PASS', panel, flush=True)


def lane(plan, panel, arch):
    assert m.read(ROOT/f'SMOKE_{panel}.json')['passed']
    for weight in [.1, 0., .03, .3]:
        for seed in plan['seeds_by_weight'][f'{weight:g}']:
            train(plan, panel, arch, weight, seed)
            evaluate(plan, panel, arch, weight, seed)
    m.write(ROOT/f'LANE_{panel}_{arch}.json', {'complete': True, 'plan': m.sha(ROOT/'PLAN.json')})


def summary(plan):
    results = {}
    for panel in ['cross', 'within']:
        results[panel] = {}
        for weight in plan['weights']:
            for seed in plan['seeds_by_weight'][f'{weight:g}']:
                train_pair = []
                for arch in ['base', 'q0']:
                    name = spec_name(arch, weight, seed)
                    trained = m.read(ROOT/'models'/panel/name/'complete.json')
                    evaluated = m.read(ROOT/'evaluation'/panel/name/'complete.json')
                    assert trained['complete'] and evaluated['complete'] and evaluated['raw_audit_passed']
                    assert evaluated['n'] == 100 and trained['updates'] == plan['panels'][panel]['updates']
                    assert trained['provenance'] == evaluated['provenance']
                    assert m.sha(trained['checkpoint']) == trained['checkpoint_sha256']
                    hits = evaluated['first_success']
                    assert evaluated['counts'] == {f's{k}':sum(x is not None and x<=k for x in hits.values())
                                                   for k in [1,3,5,9,20]}
                    train_pair.append(trained)
                    results[panel][name] = {'training': trained, 'evaluation': evaluated}
                assert train_pair[0]['draw_sha256'] == train_pair[1]['draw_sha256']
                assert train_pair[0]['initial_backbone_sha256'] == train_pair[1]['initial_backbone_sha256']
    m.write(ROOT/'SUMMARY.json', {'complete': True, 'paired_training_verified': True,
                                'plan': m.sha(ROOT/'PLAN.json'), 'splits': results})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare','smoke','lane','summary'])
    ap.add_argument('--panel', choices=['cross','within'], default='cross')
    ap.add_argument('--arch', choices=['base','q0'], default='base')
    ap.add_argument('--initialization', choices=['pretrained','scratch'], default='pretrained')
    a = ap.parse_args(); torch.set_num_threads(4)
    if a.command == 'prepare': prepare(a.initialization)
    else:
        plan = checked()
        if a.command == 'smoke': smoke(plan, a.panel)
        elif a.command == 'lane': lane(plan, a.panel, a.arch)
        else: summary(plan)
