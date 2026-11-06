"""Matched memory loss intervention following the frozen-predictor probe study.

Only memory parameters change. Original probe supplies geometry gradients;
all three adapted mixed probes are independent, frozen secondary readouts.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import AttnProbe, surface_dist
from redesign_lib import Store, load_models, context_memory, digest, write_json
from probe_adaptation import readout, save_npz
from attribution_a_metrics import describe, log_score, auc

ARMS = ['latent_only', 'latent_geometry']
HEADS = ['original'] + [f'predicted_mix_s{s}' for s in range(3)]


def provenance(root):
    return {'manifest': digest(root / 'manifest.json'),
            'code': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}}


def inputs(root, smoke=False):
    m = json.loads((root / 'manifest.json').read_text())
    prior = Path(m['probe_run'])
    assert digest(prior / 'manifest.json') == m['probe_manifest_sha256']
    if not smoke:
        result = json.loads((prior / 'SUMMARY.json').read_text())
        assert result['complete'] and result['audit_passed']
        assert result['arrays_verified'] == 129 * 40
    for path, h in [(m['base'], m['base_sha256']), (m['probe'], m['probe_sha256']),
                    (m['memory']['path'], m['memory']['sha256'])]:
        assert digest(path) == h
    return m, prior


def geometry_loss(probe, prediction, features, times):
    """Equal query weight, then equal dynamic-object/time weight; initial queries only."""
    per_query = []
    for j, (f, ts) in enumerate(zip(features, times)):
        tok = prediction[j, ts].float()
        kv = probe.inp(tok) + probe.space
        objects = []
        for o in range(f.shape[1]):
            if f[0, o, 13] >= .5 or f[0, o, 11] >= .5:
                continue
            q = torch.as_tensor(np.repeat(f[:1, o], len(ts), axis=0), device=tok.device)
            xy, angle = probe.read(kv, q)
            target = torch.as_tensor(f[ts, o, :2] * 256, device=tok.device)
            loss = (xy - target).abs().mean() / 256 * 10
            if f[0, o, 4:8].argmax() == 1:
                theta = torch.as_tensor(f[ts, o, 2] * (4 * np.pi), device=tok.device)
                loss = loss + (angle - torch.stack([theta.cos(), theta.sin()], -1)).abs().mean()
            objects.append(loss)
        assert objects
        per_query.append(torch.stack(objects).mean())
    return torch.stack(per_query).mean()


def episode(rows, seed, step):
    rng = np.random.default_rng([20261001, seed, step])
    row = rows[int(rng.integers(len(rows)))]
    slots = rng.choice(len(row['query']), min(8, len(row['query'])), replace=False).tolist()
    return row, [row['query'][s] for s in slots], [row['times'][s] for s in slots]


def frozen_fingerprint(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(root, arm, seed, smoke):
    m, prior = inputs(root, smoke); prov = provenance(root)
    folder = root / ('smoke_models' if smoke else 'models') / f'{arm}_s{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        done = json.loads((folder / 'complete.json').read_text())
        assert done['provenance'] == prov and digest(folder / 'memory.pt') == done['checkpoint_sha256']
        return
    store = Store()
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest() == m['store_files_sha256']
    for path, h in m['training_pack_meta_sha256'].items(): assert digest(path) == h
    base, probe, rule, pc, model = load_models(memory=m['memory']['path'])
    base.eval().requires_grad_(False); probe.eval().requires_grad_(False)
    fingerprints = {name: frozen_fingerprint(mod) for name, mod in [('base', base), ('probe', probe)]}
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(not p.requires_grad for p in base.parameters())
    torch.manual_seed(seed)
    cfg = m['geometry_training']; steps = 3 if smoke else cfg['steps']
    rows = [r for r in m['rows'] if r['split'] == 'train' and
            (not smoke or r['task'] in m['smoke_train_tasks'])]
    opt = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    start = 0; log = []; draws = []; checks = {}; t0 = time.time()
    resume = folder / 'resume.pt'
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        assert ck['provenance'] == prov and ck['arm'] == arm and ck['seed'] == seed and ck['steps'] == steps
        model.load_state_dict(ck['weights'], strict=False); opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); torch.set_rng_state(ck['rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start, log, draws, checks = ck['step'], ck['log'], ck['draws'], ck['checks']
    weight = cfg['geometry_weight'] if arm == 'latent_geometry' else 0.
    for step in range(start, steps):
        row, ids, times = episode(rows, seed, step)
        assert not set(ids) & set(row['context']) and len(row['context']) == 1
        assert not store.labels[row['context']].any()
        draws.append({'task': row['task'], 'query': ids, 'times': times, 'context': row['context']})
        z = torch.from_numpy(store.z(ids)).cuda().float()
        act = torch.from_numpy(store.actions[ids]).cuda(); features = [store.feature(i) for i in ids]
        model.train(); base.eval(); probe.eval()
        mem, mask = context_memory(model, store, row['context'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(z[:, 0], act, mem.expand(len(ids), -1, -1), mask.expand(len(ids), -1)).float()
        latent = (pred[:, 1:] - z[:, 1:]).abs().mean()
        geo = geometry_loss(probe, pred, features, times)
        loss = latent + weight * geo
        assert torch.isfinite(loss)
        if step == 0:
            grad = torch.autograd.grad(geo, pred, retain_graph=True)[0]
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0
            assert torch.count_nonzero(grad[:, 0]) == 0
            checks['geometry_to_prediction_grad_norm'] = float(grad.norm())
            checks['first_latent_loss'] = float(latent.detach())
            checks['first_geometry_loss'] = float(geo.detach())
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, cfg['clip'])
        assert torch.isfinite(norm) and norm > 0
        assert all(p.grad is None for p in base.parameters()) and all(p.grad is None for p in probe.parameters())
        opt.step(); scheduler.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            entry = {'step': step + 1, 'latent_l1': float(latent.detach()), 'geometry': float(geo.detach()),
                     'weighted_geometry': float(weight * geo.detach()), 'grad_norm': float(norm),
                     'seconds_this_attempt': round(time.time() - t0)}
            log.append(entry); write_json(folder / 'progress.json', entry)
            print(arm, seed, json.dumps(entry), flush=True)
        if (step + 1) % 200 == 0 or step + 1 == steps:
            weights = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('base.')}
            ck = {'provenance': prov, 'arm': arm, 'seed': seed, 'steps': steps, 'step': step + 1,
                  'weights': weights, 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                  'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                  'log': log, 'draws': draws, 'checks': checks}
            torch.save(ck, folder / 'resume.tmp'); (folder / 'resume.tmp').replace(resume)
    assert fingerprints == {name: frozen_fingerprint(mod) for name, mod in [('base', base), ('probe', probe)]}
    original = torch.load(m['memory']['path'], map_location='cpu', weights_only=False)
    assert any(not torch.equal(v, original['state_dict'][k]) for k, v in weights.items())
    ck = {key: original[key] for key in ['mode', 'mem_steps', 'mem_grid', 'per_attempt', 'base']}
    ck.update(state_dict=weights, gate_empty=True, unfreeze=False, provenance=prov, arm=arm, seed=seed, step=steps)
    torch.save(ck, folder / 'memory.tmp'); (folder / 'memory.tmp').replace(folder / 'memory.pt')
    draw_hash = hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest()
    write_json(folder / 'draws.json', draws)
    write_json(folder / 'complete.json', {'complete': True, 'provenance': prov, 'arm': arm, 'seed': seed,
               'steps': steps, 'draw_sha256': draw_hash, 'checkpoint_sha256': digest(folder / 'memory.pt'),
               'frozen_modules_unchanged': True, 'checks': checks, 'log': log})


def load_heads(m, prior, original):
    heads = {'original': original}; hashes = {'original': m['probe_sha256']}
    for name in HEADS[1:]:
        folder = prior / 'heads' / name
        done = json.loads((folder / 'complete.json').read_text())
        assert digest(folder / 'probe.pt') == done['checkpoint_sha256']
        ck = torch.load(folder / 'probe.pt', map_location='cpu', weights_only=False)
        head = AttnProbe(ck['D'], ck['S']).cuda(); head.load_state_dict(ck['state_dict'])
        heads[name] = head.eval().requires_grad_(False); hashes[name] = done['checkpoint_sha256']
    return heads, hashes


@torch.no_grad()
def evaluate(root, arm, seed, smoke):
    m, prior = inputs(root, smoke); prov = provenance(root)
    name = f'{arm}_s{seed}'; model_dir = root / ('smoke_models' if smoke else 'models') / name
    ck = json.loads((model_dir / 'complete.json').read_text())
    assert ck['provenance'] == prov and digest(model_dir / 'memory.pt') == ck['checkpoint_sha256']
    base, original, rule, pc, model = load_models(memory=model_dir / 'memory.pt')
    heads, hashes = load_heads(m, prior, original)
    prov = {**prov, 'checkpoint': ck['checkpoint_sha256'], 'heads': hashes}
    stage = json.loads((Path(m['stage_a']) / 'manifest.json').read_text())
    assert digest(Path(m['stage_a']) / 'manifest.json') == m['stage_a_manifest_sha256']
    store = Store(); lat = np.load(stage['pack'] + '_latents.npy', mmap_mode='r')
    assert digest(stage['pack'] + '_meta.npz') == stage['meta_sha256']
    with np.load(stage['pack'] + '_meta.npz') as packed: meta = {k: packed[k] for k in packed.files}
    entries = [('cross', r) for r in stage['tasks'] if not smoke or r['task'] in m['smoke_cross_tasks']]
    if not smoke: entries += [('calibration', r) for r in m['rows'] if r['split'] == 'calibration']
    out = root / ('smoke_eval' if smoke else 'evaluation') / name; t0 = time.time()
    for panel, entry in entries:
        task = entry['task']; folder = out / panel / task.replace(':', '_'); folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'summary.json').exists():
            assert json.loads((folder / 'summary.json').read_text())['provenance'] == prov
            continue
        reference = prior / 'evaluation' / panel / task.replace(':', '_')
        ref = json.loads((reference / 'summary.json').read_text())
        if panel == 'cross':
            ids = entry['indices']; labels = meta['solved'][ids].astype(bool); mask = np.isin(ids, entry['query'])
            features = [meta['features'][i][:, meta['mask'][i]] for i in ids]; context = entry['contexts']['K1']
            zfun = lambda ii: np.stack([lat[i] for i in ii]); afun = lambda ii: meta['action'][ii]
            k = max(1, len(context)); cz = torch.zeros((1, k) + lat.shape[1:], device='cuda')
            ca = torch.zeros(1, k, 3, device='cuda'); cy = torch.zeros(1, k, device='cuda'); cm = torch.zeros(1, k, dtype=torch.bool, device='cuda')
            for j, i in enumerate(context):
                assert not meta['solved'][i]; cz[0, j] = torch.from_numpy(np.array(lat[i])).cuda().float()
                ca[0, j] = torch.from_numpy(meta['action'][i]).cuda(); cm[0, j] = True
            with torch.autocast('cuda', dtype=torch.bfloat16):
                mem, mm = model.encode_memory(cz, ca, cy, cm, model.context_residual(cz, ca).float())
        else:
            ids = entry['query']; labels = store.labels[ids]; mask = np.ones(len(ids), bool)
            features = [store.feature(i) for i in ids]; context = entry['context']
            zfun = store.z; afun = lambda ii: store.actions[ii]; mem, mm = context_memory(model, store, context)
        truth = {'distance': np.stack([surface_dist(f) for f in features]), 'xy': np.stack([f[:, :, :2] * 256 for f in features]),
                 'labels': labels, 'query_mask': mask, 'indices': np.asarray(ids)}
        with np.load(reference / 'truth.npz') as v:
            for key in truth: np.testing.assert_array_equal(truth[key], v[key])
        collected = {h: {} for h in heads}; errors = []; squared = []; base_replay = []; boundary = False
        for start in range(0, len(ids), 32):
            ii = ids[start:start + 32]; fs = features[start:start + 32]
            z = torch.from_numpy(zfun(ii)).cuda().float(); act = torch.from_numpy(afun(ii)).cuda(); z0 = z[:, 0].contiguous()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                bp = base(z0, act).float(); pred = model(z0, act, mem.expand(len(ii), -1, -1), mm.expand(len(ii), -1)).float()
            assert torch.equal(pred[:, 0], z0)
            if not context: assert torch.equal(bp, pred)
            diff = pred[:, 1:] - z[:, 1:]; errors.append(diff.abs().mean((1, 2, 3)).cpu().numpy())
            squared.append(diff.square().mean((1, 2, 3)).sqrt().cpu().numpy())
            base_replay.append(readout(original, bp, fs, rule)['log_score'])
            for head_name, head in heads.items():
                value = readout(head, pred, fs, rule)
                for key, v in value.items(): collected[head_name].setdefault(key, []).append(v)
                if not boundary:
                    altered = [f.copy() for f in fs]
                    for f in altered: f[1:] = 123
                    changed = readout(head, pred, altered, rule)
                    for key in value: np.testing.assert_array_equal(value[key], changed[key])
            boundary = True
        with np.load(reference / 'original__base.npz') as v:
            np.testing.assert_array_equal(np.concatenate(base_replay), v['log_score'])
        save_npz(folder / 'truth.npz', truth); metrics = {}
        for head_name, fields in collected.items():
            values = {k: np.concatenate(v) for k, v in fields.items()}
            values['latent_l1'] = np.concatenate(errors); values['latent_l2_rms'] = np.concatenate(squared)
            save_npz(folder / (head_name + '.npz'), values)
            metrics[head_name] = describe(values, truth, labels, mask, rule, ref['green'], ref['dynamic_goals'])
        write_json(folder / 'summary.json', {'task': task, 'panel': panel, 'provenance': prov, 'model': name,
                   'context': context, 'query': np.asarray(ids)[mask].tolist(), 'conditions': metrics,
                   'rule': rule, 'green': ref['green'], 'dynamic_goals': ref['dynamic_goals'],
                   'base_exact_replay': True, 'empty_identity': True, 'future_input_isolation': boundary,
                   'reference_summary_sha256': digest(reference / 'summary.json')})
        print(name, panel, task, round(time.time() - t0), flush=True)
    write_json(out / 'complete.json', {'complete': True, 'provenance': prov, 'tasks': len(entries)})


def summarize(root, smoke):
    m, prior = inputs(root, smoke); prov = provenance(root); seeds = [0] if smoke else m['seeds']
    stage = json.loads((Path(m['stage_a']) / 'manifest.json').read_text())
    entries = [('cross', r['task']) for r in stage['tasks'] if not smoke or r['task'] in m['smoke_cross_tasks']]
    if not smoke: entries += [('calibration', r['task']) for r in m['rows'] if r['split'] == 'calibration']
    rows = {}; count = 0
    for seed in seeds:
        checks = []
        for arm in ARMS:
            name = f'{arm}_s{seed}'; folder = root / ('smoke_models' if smoke else 'models') / name
            done = json.loads((folder / 'complete.json').read_text())
            assert done['provenance'] == prov and done['frozen_modules_unchanged']
            assert digest(folder / 'memory.pt') == done['checkpoint_sha256']; checks.append(done)
            rows[name] = []
            for panel, task in entries:
                d = root / ('smoke_eval' if smoke else 'evaluation') / name / panel / task.replace(':', '_')
                row = json.loads((d / 'summary.json').read_text())
                assert row['task'] == task and row['panel'] == panel
                assert row['provenance']['manifest'] == prov['manifest'] and row['provenance']['code'] == prov['code']
                assert row['provenance']['checkpoint'] == done['checkpoint_sha256']
                assert row['base_exact_replay'] and row['empty_identity'] and row['future_input_isolation']
                refpath = prior / 'evaluation' / panel / task.replace(':', '_') / 'summary.json'
                assert digest(refpath) == row['reference_summary_sha256']
                reference = json.loads(refpath.read_text())
                with np.load(d / 'truth.npz') as truth:
                    assert truth['indices'][truth['query_mask']].tolist() == row['query']
                    for h in HEADS:
                        with np.load(d / (h + '.npz')) as v:
                            np.testing.assert_allclose(log_score(v['distance'], row['rule']), v['log_score'], rtol=1e-12, atol=1e-12)
                            computed = describe(v, truth, truth['labels'], truth['query_mask'], row['rule'], row['green'], row['dynamic_goals'])
                            assert computed == row['conditions'][h]
                            assert computed['all']['latent_l1'] == row['conditions']['original']['all']['latent_l1']
                        count += 1
                row['reference'] = reference['conditions']; rows[name].append(row)
        assert checks[0]['draw_sha256'] == checks[1]['draw_sha256']
        assert checks[0]['checks']['first_latent_loss'] == checks[1]['checks']['first_latent_loss']
        assert checks[0]['checks']['first_geometry_loss'] == checks[1]['checks']['first_geometry_loss']
    if smoke:
        write_json(root / 'smoke_check.json', {'passed': True, 'arrays_verified': count, 'provenance': prov,
                   'scope': 'Engineering smoke only; 3-step updates, no scientific selection.'})
        print('SMOKE_PASSED', count, flush=True); return
    def get(c, key):
        for part in key.split('.'): c = c[part]
        return c
    def avg(v):
        v = [x for x in v if x is not None]
        return {'mean': float(np.mean(v)) if v else None, 'n': len(v)}
    metrics = ['auc', 'hit_at_10', 'all.latent_l1', 'success.latent_l1', 'all.surface_mae', 'success.surface_mae', 'success.green_position_error']
    panels = {}
    for panel in ['cross_nonempty_K1', 'cross_all100', 'calibration']:
        select = lambda r: (r['panel'] == 'calibration') if panel == 'calibration' else (r['panel'] == 'cross' and (panel == 'cross_all100' or bool(r['context'])))
        macros = {}; paired = {}
        for name, rr in rows.items():
            selected = [r for r in rr if select(r)]
            macros[name] = {h: {key: avg([get(r['conditions'][h], key) for r in selected]) for key in metrics} for h in HEADS}
        for seed in seeds:
            a = [r for r in rows[f'latent_only_s{seed}'] if select(r)]
            b = [r for r in rows[f'latent_geometry_s{seed}'] if select(r)]
            assert [(r['panel'], r['task']) for r in a] == [(r['panel'], r['task']) for r in b]
            paired[str(seed)] = {}
            for h in HEADS:
                records = []
                for x, y in zip(a, b):
                    rec = {'task': x['task'], 'template': x['task'][:5]}
                    for key in metrics:
                        v = [get(x['conditions'][h], key), get(y['conditions'][h], key),
                             get(x['reference'][h + '__memory'], key), get(x['reference'][h + '__base'], key)]
                        rec[key] = None if any(z is None for z in v) else {'geometry_minus_l1': float(v[1]) - float(v[0]),
                            'geometry_minus_R0': float(v[1]) - float(v[2]), 'geometry_minus_base': float(v[1]) - float(v[3])}
                    records.append(rec)
                summaries = {}
                for key in metrics:
                    good = [r for r in records if r[key] is not None]
                    summaries[key] = {c: avg([r[key][c] for r in good]) for c in ['geometry_minus_l1', 'geometry_minus_R0', 'geometry_minus_base']}
                    summaries[key]['by_template'] = {t: {c: avg([r[key][c] for r in good if r['template'] == t]) for c in ['geometry_minus_l1', 'geometry_minus_R0', 'geometry_minus_base']} for t in sorted({r['template'] for r in good})}
                paired[str(seed)][h] = {'summary': summaries, 'per_task': records}
        panels[panel] = {'macros': macros, 'paired': paired}
    # Fixed, exploratory continuation criterion. It is not a significance test.
    gate = {}
    for panel in ['cross_nonempty_K1', 'calibration']:
        signs = []
        for seed in seeds:
            heads = panels[panel]['paired'][str(seed)]
            main = heads['original']['summary']
            original_ok = main['auc']['geometry_minus_l1']['mean'] > 0 and main['success.surface_mae']['geometry_minus_l1']['mean'] < 0
            secondary_auc = np.mean([heads[h]['summary']['auc']['geometry_minus_l1']['mean'] for h in HEADS[1:]])
            secondary_geo = np.mean([heads[h]['summary']['success.surface_mae']['geometry_minus_l1']['mean'] for h in HEADS[1:]])
            signs.append({'seed': seed, 'original_both_improve': bool(original_ok), 'adapted_mean_both_improve': bool(secondary_auc > 0 and secondary_geo < 0)})
        gate[panel] = signs
    proceed = all(sum(s['original_both_improve'] and s['adapted_mean_both_improve'] for s in signs) >= 2 for signs in gate.values())
    result = {'complete': True, 'audit_passed': True, 'arrays_verified': count, 'provenance': prov,
              'probe_summary_sha256': digest(prior / 'SUMMARY.json'), 'panels': panels,
              'task_latent_next': {'criterion_passed': proceed, 'evidence': gate,
                'status': 'authorized_for_next_stage_design' if proceed else 'diagnose_geometry_intervention_before_representation_redesign',
                'scope': 'Exploratory continuation gate only; reused D0 and 3 seeds from shared R0, no significance/generalization claim. No automatic representation-training launch.'}}
    write_json(root / 'SUMMARY.json', result)
    lines = ['# Memory geometry-loss intervention', '', 'Fixed original training probe; paired R0 initialization/data/steps. No closed-loop result.', '',
             '| Model | Readout | AUC | Successful surface MAE (px) |', '|---|---|---:|---:|']
    for name, heads in panels['cross_nonempty_K1']['macros'].items():
        for h, mm in heads.items(): lines.append(f"| {name} | {h} | {mm['auc']['mean']:.5f} | {mm['success.surface_mae']['mean']:.3f} |")
    lines += ['', f'Task-latent continuation criterion passed: {proceed}. See SUMMARY.json for every denominator and paired/template effect.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('SUMMARY_COMPLETE', count, 'task_latent_criterion', proceed, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['train', 'eval', 'summary'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--arm', choices=ARMS, default=ARMS[0])
    p.add_argument('--seed', type=int, default=0); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); torch.set_num_threads(4)
    if a.stage == 'train': train(a.root, a.arm, a.seed, a.smoke)
    elif a.stage == 'eval': evaluate(a.root, a.arm, a.seed, a.smoke)
    else: summarize(a.root, a.smoke)


if __name__ == '__main__': main()
