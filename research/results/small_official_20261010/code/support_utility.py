"""Matched support-sampling experiment. Futures are targets, never query inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from common import surface_dist, longest
from redesign_lib import Store, load_models, context_memory, digest, write_json, ROOT, PROJECT
from probe_adaptation import readout, save_npz
from memory_geometry_followup import geometry_loss, frozen_fingerprint, load_heads
from attribution_a_metrics import describe

ARMS = ['uniform', 'near', 'mixed']
SEED = 20261004
PRIOR = PROJECT / 'results/memory_geometry_followup_20261001'
PROBES = PROJECT / 'results/probe_adaptation_cross_20261001'


def read(path):
    return json.loads(Path(path).read_text())


def stable_int(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16], 16)


def grouped_partition(store, task, ids, clusters):
    """Union repeated actions and recorded cluster membership before role split."""
    parent = {int(i): int(i) for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def join(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    by_group = defaultdict(list)
    for i in ids:
        i = int(i)
        known = clusters.get(str(store.files[i]), [])
        keys = [('cluster', c) for c in known] if known else [('grid', *np.floor(store.actions[i] / .02).astype(int).tolist())]
        for key in keys: by_group[key].append(i)
    for members in by_group.values():
        for i in members[1:]: join(members[0], i)
    groups = defaultdict(list)
    for i in ids: groups[find(int(i))].append(int(i))
    roles = {'discovery': [], 'support': [], 'query': []}; assignment = {}
    for members in groups.values():
        signature = ','.join(sorted(str(store.files[i]) for i in members))
        u = stable_int(f'{SEED}:{task}:{signature}') / 2**64
        role = 'discovery' if u < .2 else 'support' if u < .6 else 'query'
        roles[role].extend(members)
        for i in members: assignment[str(i)] = {'group': min(members), 'role': role}
    return {k: sorted(v) for k, v in roles.items()}, assignment


def inventory(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'inventory.json').exists(): raise RuntimeError('Inventory already exists; do not silently replace splits')
    store = Store(); old = read(PROJECT / 'results/memory_redesign_20260928/data/dataset.json')
    source = PROJECT / 'data/manifests/ball_cross_template_f1_train_clusters.jsonl'
    clusters = defaultdict(list)
    for line in source.read_text().splitlines():
        r = json.loads(line)
        if 'cluster' in r: clusters[r['file']].append(int(r['cluster']))
    # Existing Store deduplicates filenames; additionally exclude identical task/action aliases.
    seen = {}; aliases = {}; ids_by = defaultdict(list)
    for i in range(len(store.files)):
        key = (str(store.tasks[i]), tuple(store.actions[i].tolist()))
        if key in seen:
            j = seen[key]; assert store.labels[i] == store.labels[j]
            aliases[str(i)] = j
            clusters[str(store.files[j])].extend(clusters.get(str(store.files[i]), []))
        else: seen[key] = i; ids_by[key[0]].append(i)
    train = set(old['train_tasks']); held = set(old['memory_holdout_templates'])
    rows = []; t0 = time.time()
    for task in sorted(ids_by):
        ids = ids_by[task]
        split = 'train' if task in train else 'calibration'
        assert split == 'train' or task[:5] in held
        roles, groups = grouped_partition(store, task, ids, clusters)
        quality = {}
        for i in ids:
            d = surface_dist(store.feature(i)); assert len(d) == 32 and np.isfinite(d).all()
            quality[str(i)] = {'minimum': float(d[1:].min()), 'initial': float(d[0]),
                'near_fraction': float((d[1:] <= 5).mean()), 'near_longest': longest(d[1:] <= 5)}
        failures = [i for i in roles['support'] if not store.labels[i]]
        near = [i for i in failures if quality[str(i)]['minimum'] <= 5]
        far = [i for i in failures if quality[str(i)]['minimum'] >= 20]
        ds = roles['discovery']; anchors = [i for i in ds if store.labels[i]]
        nearest = {}
        if anchors:
            for i in roles['support']:
                nearest[str(i)] = float(np.linalg.norm(store.actions[anchors] - store.actions[i], axis=1).min())
        sq = None
        if roles['support'] and roles['query']:
            a = store.actions[roles['support']]; b = store.actions[roles['query']]
            sq = float(np.linalg.norm(a[:, None] - b[None, :], axis=-1).min())
        row = {'task': task, 'split': split, **roles, 'groups': groups, 'quality': quality,
            'failures': failures, 'N': near, 'far': far, 'anchor_distance': nearest,
            'support_query_min_action_distance': sq, 'query_successes': int(store.labels[roles['query']].sum()),
            'potential': bool(near and far and roles['query'])}
        rows.append(row)
    prior = read(PRIOR / 'manifest.json')
    stage = read(Path(prior['stage_a']) / 'manifest.json')
    summary = {}
    for split in ['train', 'calibration']:
        rr = [r for r in rows if r['split'] == split]
        summary[split] = {'scenes': len(rr), 'potential_scenes': sum(r['potential'] for r in rr),
            'potential_by_template': dict(Counter(r['task'][:5] for r in rr if r['potential'])),
            'support_actions_to_score': sum(len(r['support']) for r in rr if r['potential']),
            'no_query_scenes': sum(not r['query'] for r in rr),
            'N_initial_near': sum(r['quality'][str(i)]['initial'] <= 5 for r in rr for i in r['N']),
            'N_count': sum(len(r['N']) for r in rr)}
    result = {'seed': SEED, 'rows': rows, 'summary': summary, 'aliases': aliases,
        'store_files_sha256': hashlib.sha256('\n'.join(store.files).encode()).hexdigest(),
        'source_hashes': {str(source): digest(source), **prior['training_pack_meta_sha256']},
        'prior_manifest_sha256': digest(PRIOR / 'manifest.json'),
        'checkpoints': {k: prior[k] for k in ['base', 'base_sha256', 'probe', 'probe_sha256', 'memory']},
        'probe_run': str(PROBES), 'stage_a': prior['stage_a'], 'stage_a_manifest_sha256': digest(Path(prior['stage_a']) / 'manifest.json'),
        'cross_tasks': stage['tasks'], 'cross_pack': stage['pack'], 'cross_meta_sha256': stage['meta_sha256'],
        'seconds': time.time() - t0}
    write_json(root / 'inventory.json', result); write_json(root / 'COVERAGE_PRE_SCORE.json', summary)
    print(json.dumps(summary), flush=True)


@torch.no_grad()
def score_support(root, shard, nshards):
    inv = read(root / 'inventory.json'); store = Store()
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest() == inv['store_files_sha256']
    base, original, rule, _, _ = load_models()
    heads, hh = load_heads(inv['checkpoints'], PROBES, original); head = heads['predicted_mix_s0']
    prov = {'inventory': digest(root / 'inventory.json'), 'base': inv['checkpoints']['base_sha256'], 'head': hh['predicted_mix_s0']}
    assert digest(inv['checkpoints']['base']) == prov['base']
    rows = [r for r in inv['rows'] if r['potential']][shard::nshards]
    t0 = time.time()
    for row in rows:
        out = root / 'support_scores' / (row['task'].replace(':', '_') + '.json')
        if out.exists(): assert read(out)['provenance'] == prov; continue
        ids = row['support']; scores = []
        for start in range(0, len(ids), 32):
            ii = ids[start:start+32]; z0 = torch.from_numpy(store.z(ii)[:, 0].copy()).cuda().float()
            act = torch.from_numpy(store.actions[ii]).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16): pred = base(z0, act).float()
            fs = [store.feature(i) for i in ii]
            scores.extend(readout(head, pred, fs, rule)['log_score'].tolist())
        assert np.isfinite(scores).all()
        write_json(out, {'task': row['task'], 'indices': ids, 'scores': scores, 'provenance': prov})
        print('SCORED', row['task'], len(ids), round(time.time()-t0), flush=True)
    write_json(root / f'scores_complete_{shard}.json', {'provenance': prov, 'tasks': [r['task'] for r in rows], 'nshards': nshards})


class CrossStore:
    def __init__(self, pack):
        self.lat = [np.load(pack + '_latents.npy', mmap_mode='r')]
        with np.load(pack + '_meta.npz') as p: self.meta = {k: p[k] for k in p.files}
        self.tasks = self.meta['tasks']; self.actions = self.meta['action']
        self.labels = self.meta['solved'].astype(bool); self.files = self.meta['files']
        self.by = {t: np.flatnonzero(self.tasks == t).tolist() for t in sorted(set(self.tasks))}
    def z(self, ids): return np.array(self.lat[0][ids])
    def feature(self, i): return self.meta['features'][i][:, self.meta['mask'][i]]


def classify(row, scores, labels):
    ids = row['support']; assert len(ids) == len(scores)
    order = np.argsort(-np.asarray(scores), kind='stable')
    high = {ids[j] for j in order[:math.ceil(.2 * len(ids))]}
    row['H'] = [i for i in row['far'] if i in high]
    row['F'] = [i for i in row['far'] if i not in high]
    row['eligible'] = bool(row['N'] and row['H'] and row['F'] and row['query'])
    row['top20_support'] = sorted(high)
    assert not set(row['support']) & set(row['query'])
    for category in ['N', 'H', 'F']:
        assert all(not labels[i] for i in row[category])
    assert not set(row['N']) & set(row['H'] + row['F'])
    return row


def finalize(root):
    assert not (root / 'manifest.json').exists(), 'Do not replace registered manifest'
    inv = read(root / 'inventory.json'); store = Store()
    for shard in range(2):
        done = read(root / f'scores_complete_{shard}.json')
        assert done['provenance']['inventory'] == digest(root / 'inventory.json') and done['nshards'] == 2
    rows = []
    for row in inv['rows']:
        if not row['potential']: continue
        path = root / 'support_scores' / (row['task'].replace(':', '_') + '.json')
        d = read(path); assert d['indices'] == row['support']
        assert d['provenance'] == done['provenance']
        r = classify(row, d['scores'], store.labels)
        r['score_sha256'] = digest(path)
        r['times'] = [sorted(np.random.default_rng([SEED, i, 73]).choice(np.arange(1, 32), 8, replace=False).tolist()) for i in r['query']]
        rows.append(r)
    train = [r for r in rows if r['eligible'] and r['split'] == 'train']
    coverage = {'pre_score': inv['summary']}
    for split in ['train', 'calibration']:
        rr = [r for r in rows if r['eligible'] and r['split'] == split]
        coverage[split] = {'eligible_scenes': len(rr), 'by_template': dict(Counter(r['task'][:5] for r in rr)),
            'query_actions': sum(len(r['query']) for r in rr), 'query_successes': sum(r['query_successes'] for r in rr),
            'no_success_query_scenes': sum(r['query_successes'] == 0 for r in rr),
            'support_counts': {c: sum(len(r[c]) for r in rr) for c in ['N', 'H', 'F', 'failures']}}
    write_json(root / 'COVERAGE.json', coverage)
    # This is a data feasibility guard, NOT a performance gate.
    assert len(train) >= 32 and len({r['task'][:5] for r in train}) >= 4, coverage
    assert digest(inv['cross_pack'] + '_meta.npz') == inv['cross_meta_sha256']
    cross = CrossStore(inv['cross_pack']); crossrows = []
    for task in sorted(cross.by):
        ids = cross.by[task]; roles, groups = grouped_partition(cross, task, ids, {})
        basefile = PROBES / 'evaluation/cross' / task.replace(':', '_') / 'predicted_mix_s0__base.npz'
        with np.load(basefile) as b: scores = b['log_score'].copy()
        ref = read(basefile.parent / 'summary.json'); assert task == ref['task'] and len(ids) == len(scores) == 1024
        failures = [i for i in roles['support'] if not cross.labels[i]]
        minima = {i: float(surface_dist(cross.feature(i))[1:].min()) for i in failures}
        row = {'task': task, 'split': 'cross', **roles, 'groups': groups,
            'indices': ids, 'failures': failures,
            'N': [i for i in failures if minima[i] <= 5], 'far': [i for i in failures if minima[i] >= 20]}
        row = classify(row, [scores[ids.index(i)] for i in roles['support']], cross.labels)
        first = ids[int(np.argmax(scores))]
        row['natural_context'] = [] if cross.labels[first] else [first]
        row['first_action_success'] = bool(cross.labels[first])
        row['base_scores_sha256'] = digest(basefile)
        crossrows.append(row)
    coverage['cross'] = {'scenes': len(crossrows), 'selected_eligible': sum(r['eligible'] for r in crossrows),
        'selected_by_template': dict(Counter(r['task'][:5] for r in crossrows if r['eligible'])),
        'natural_first_failed': sum(bool(r['natural_context']) for r in crossrows)}
    write_json(root / 'COVERAGE.json', coverage)
    calibration = [r for r in rows if r['eligible'] and r['split'] == 'calibration']
    # Predefined bounded calibration panel, selected without query labels/effects.
    calibration = [r for tp in sorted({r['task'][:5] for r in calibration})
        for r in sorted([r for r in calibration if r['task'][:5] == tp], key=lambda r: stable_int(f'{SEED}:cal:{r["task"]}'))[:16]]
    smoke_train = []
    for tp in sorted({r['task'][:5] for r in train})[:4]:
        smoke_train.append(next(r['task'] for r in train if r['task'][:5] == tp))
    smoke_cross = [next(r['task'] for r in crossrows if r['task'][:5] == tp) for tp in sorted({r['task'][:5] for r in crossrows})]
    cfg = read(PRIOR / 'manifest.json')['geometry_training']
    m = {**inv['checkpoints'], 'scope': 'K1 support-sampling pilot; reused D0, no fresh D1 or closed-loop claim',
        'inventory_sha256': digest(root / 'inventory.json'), 'store_files_sha256': inv['store_files_sha256'],
        'source_hashes': inv['source_hashes'], 'rows': train + calibration, 'cross_rows': crossrows,
        'cross_pack': inv['cross_pack'], 'cross_meta_sha256': inv['cross_meta_sha256'],
        'probe_run': str(PROBES), 'geometry_training': cfg, 'arms': ARMS, 'seeds': [0, 1, 2],
        'smoke_train_tasks': smoke_train, 'smoke_cross_tasks': smoke_cross,
        'heads': {name: digest(PROBES / 'heads' / name / 'probe.pt') for name in ['predicted_mix_s0', 'predicted_mix_s1', 'predicted_mix_s2']},
        'primary_head': 'predicted_mix_s0', 'donors_per_category': 1, 'coverage': coverage,
        'registration_choices': {'minimum_training_scenes': 32, 'minimum_training_templates': 4,
            'calibration_max_per_template': 16, 'near_px': 5, 'far_px': 20, 'base_top_fraction': .2,
            'primary_comparisons': ['near-uniform', 'mixed-uniform', 'mixed-near'],
            'fresh_D1': 'separate subsequent data collection; NOT part of this D0 pilot',
            'K4': 'subsequent conditional branch; not part of this K1 pilot'}}
    write_json(root / 'manifest.json', m); print(json.dumps(coverage), flush=True)


def provenance(root):
    return {'manifest': digest(root / 'manifest.json'),
        'code': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}}


def episode(rows, seed, step, arm):
    # Arm has no influence on the query/time RNG.
    rng = np.random.default_rng([SEED, seed, step, 0])
    row = rows[int(rng.integers(len(rows)))]; slots = rng.choice(len(row['query']), min(8, len(row['query'])), replace=False).tolist()
    ids = [row['query'][j] for j in slots]; times = [row['times'][j] for j in slots]
    srng = np.random.default_rng([SEED, seed, step, 1])
    category = 'failures' if arm == 'uniform' else 'N' if arm == 'near' else ['N', 'H', 'F'][int(srng.integers(3))]
    context = [int(srng.choice(row[category]))]
    return row, ids, times, context, category


def train(root, arm, seed, smoke):
    m = read(root / 'manifest.json'); prov = provenance(root)
    if not smoke:
        check = read(root / 'smoke_check.json'); assert check['passed'] and check['provenance'] == prov
    folder = root / ('smoke_models' if smoke else 'models') / f'{arm}_s{seed}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        d = read(folder / 'complete.json'); assert d['provenance'] == prov and digest(folder / 'memory.pt') == d['checkpoint_sha256']; return
    store = Store(); assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest() == m['store_files_sha256']
    for path, h in m['source_hashes'].items(): assert digest(path) == h
    for path, h in [(m['base'], m['base_sha256']), (m['probe'], m['probe_sha256']), (m['memory']['path'], m['memory']['sha256'])]: assert digest(path) == h
    base, probe, _, _, model = load_models(memory=m['memory']['path'])
    base.eval().requires_grad_(False); probe.eval().requires_grad_(False)
    fingerprints = {'base': frozen_fingerprint(base), 'probe': frozen_fingerprint(probe)}
    initial = frozen_fingerprint(model); torch.manual_seed(seed)
    cfg = m['geometry_training']; steps = 3 if smoke else cfg['steps']
    rows = [r for r in m['rows'] if r['split'] == 'train' and (not smoke or r['task'] in m['smoke_train_tasks'])]
    params = [p for p in model.parameters() if p.requires_grad]; assert params
    opt = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    start = 0; draws = []; log = []; checks = {'initial_model_sha256': initial}; resume = folder / 'resume.pt'; t0 = time.time()
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        assert ck['provenance'] == prov and ck['arm'] == arm and ck['seed'] == seed and ck['steps'] == steps
        model.load_state_dict(ck['weights'], strict=False); opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start, draws, log, checks = ck['step'], ck['draws'], ck['log'], ck['checks']
    for step in range(start, steps):
        row, ids, times, context, category = episode(rows, seed, step, arm)
        assert not set(ids) & set(context) and len(context) == 1 and not store.labels[context].any()
        assert all(store.tasks[i] == row['task'] for i in ids + context)
        draws.append({'task': row['task'], 'query': ids, 'times': times, 'context': context, 'category': category,
            'query_successes': int(store.labels[ids].sum())})
        z = torch.from_numpy(store.z(ids)).cuda().float(); act = torch.from_numpy(store.actions[ids]).cuda()
        features = [store.feature(i) for i in ids]
        if step == 0:
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                em, mask = context_memory(model, store, [])
                bp = base(z[:, 0], act).float(); ep = model(z[:, 0], act, em.expand(len(ids), -1, -1), mask.expand(len(ids), -1)).float()
                assert torch.equal(bp, ep)
                # Common full support input for all arms proves equal initialized functions.
                cm, mask = context_memory(model, store, [row['N'][0]])
                cp = model(z[:, 0], act, cm.expand(len(ids), -1, -1), mask.expand(len(ids), -1)).float()
                checks['common_input_output_sha256'] = hashlib.sha256(cp.cpu().numpy().tobytes()).hexdigest()
            checks['empty_identity'] = True
        model.train(); base.eval(); probe.eval()
        mem, mask = context_memory(model, store, context)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(z[:, 0], act, mem.expand(len(ids), -1, -1), mask.expand(len(ids), -1)).float()
        latent = (pred[:, 1:] - z[:, 1:]).abs().mean(); geo = geometry_loss(probe, pred, features, times)
        loss = latent + cfg['geometry_weight'] * geo; assert torch.isfinite(loss)
        if step == 0:
            grad = torch.autograd.grad(geo, pred, retain_graph=True)[0]
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0 and torch.count_nonzero(grad[:, 0]) == 0
            checks['geometry_gradient_nonzero'] = True
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, cfg['clip']); assert torch.isfinite(norm) and norm > 0
        assert all(p.grad is None for p in base.parameters()) and all(p.grad is None for p in probe.parameters())
        opt.step(); scheduler.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            rec = {'step': step + 1, 'latent_l1': float(latent.detach()), 'geometry': float(geo.detach()), 'seconds': time.time()-t0}
            log.append(rec); write_json(folder / 'progress.json', rec); print(arm, seed, json.dumps(rec), flush=True)
        if (step + 1) % 200 == 0 or step + 1 == steps:
            weights = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('base.')}
            ck = {'provenance': prov, 'arm': arm, 'seed': seed, 'steps': steps, 'step': step+1, 'weights': weights,
                'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'draws': draws, 'log': log, 'checks': checks}
            torch.save(ck, folder / 'resume.tmp'); (folder / 'resume.tmp').replace(resume)
    assert fingerprints == {'base': frozen_fingerprint(base), 'probe': frozen_fingerprint(probe)}
    assert frozen_fingerprint(model) != initial
    original = torch.load(m['memory']['path'], map_location='cpu', weights_only=False)
    ck = {k: original[k] for k in ['mode', 'mem_steps', 'mem_grid', 'per_attempt', 'base']}
    ck.update(state_dict=weights, gate_empty=True, unfreeze=False, arm=arm, seed=seed, step=steps, provenance=prov)
    torch.save(ck, folder / 'memory.tmp'); (folder / 'memory.tmp').replace(folder / 'memory.pt')
    common = [{k: r[k] for k in ['task', 'query', 'times']} for r in draws]
    write_json(folder / 'draws.json', draws)
    write_json(folder / 'complete.json', {'complete': True, 'provenance': prov, 'arm': arm, 'seed': seed,
        'steps': steps, 'common_draw_sha256': hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest(),
        'checkpoint_sha256': digest(folder / 'memory.pt'), 'frozen_modules_unchanged': True, 'checks': checks,
        'support_categories': dict(Counter(r['category'] for r in draws)),
        'unique_supports': len({i for r in draws for i in r['context']}),
        'no_success_query_steps': sum(r['query_successes'] == 0 for r in draws), 'log': log})


def eval_specs(m, smoke):
    specs = []
    for row in m['cross_rows']:
        if smoke and row['task'] not in m['smoke_cross_tasks']: continue
        specs.append(('cross_natural', row, 'natural', row['natural_context']))
        if row['eligible']:
            for cat in ['N', 'H', 'F']:
                donor = min(row[cat], key=lambda i: stable_int(f'{SEED}:donor:{row["task"]}:{i}'))
                specs.append(('cross_selected', row, cat, [donor]))
    if not smoke:
        for row in m['rows']:
            if row['split'] != 'calibration': continue
            for cat in ['N', 'H', 'F']:
                donor = min(row[cat], key=lambda i: stable_int(f'{SEED}:donor:{row["task"]}:{i}'))
                specs.append(('calibration_selected', row, cat, [donor]))
    return specs


@torch.no_grad()
def evaluate(root, arm, seed, smoke):
    m = read(root / 'manifest.json'); prov = provenance(root); name = f'{arm}_s{seed}'
    modeldir = root / ('smoke_models' if smoke else 'models') / name; complete = read(modeldir / 'complete.json')
    assert complete['provenance'] == prov and digest(modeldir / 'memory.pt') == complete['checkpoint_sha256']
    base, original, rule, _, model = load_models(memory=modeldir / 'memory.pt')
    model.eval(); base.eval(); original.eval()
    heads, hh = load_heads(m, PROBES, original)
    assert all(hh[h] == sha for h, sha in m['heads'].items())
    prov = {**prov, 'checkpoint': complete['checkpoint_sha256'], 'heads': hh}
    cross = CrossStore(m['cross_pack']); store = Store()
    assert digest(m['cross_pack'] + '_meta.npz') == m['cross_meta_sha256']
    out = root / ('smoke_eval' if smoke else 'evaluation') / name; t0 = time.time(); specs = eval_specs(m, smoke)
    for panel, row, condition, context in specs:
        task = row['task']; folder = out / panel / task.replace(':', '_') / condition
        if (folder / 'summary.json').exists(): assert read(folder / 'summary.json')['provenance'] == prov; continue
        folder.mkdir(parents=True, exist_ok=True)
        source = cross if panel.startswith('cross') else store
        ids = row['indices'] if panel.startswith('cross') else row['query']
        # Always preserve historical 32-sized batches for exact cross-base replay.
        query = [i for i in ids if i not in context] if condition == 'natural' else row['query']
        mask = np.isin(ids, query); labels = source.labels[ids]
        assert not set(query) & set(context) and not source.labels[context].any()
        fs = [source.feature(i) for i in ids]
        green = int(np.flatnonzero(fs[0][0, :, 9] > .5)[0])
        goals = [o for o in range(fs[0].shape[1]) if fs[0][0, o, 10] > .5 and fs[0][0, o, 13] < .5 and fs[0][0, o, 11] < .5]
        truth = {'indices': np.asarray(ids), 'labels': labels, 'query_mask': mask,
            'distance': np.stack([surface_dist(f) for f in fs]), 'xy': np.stack([f[:, :, :2]*256 for f in fs])}
        cached_base = {}
        if panel.startswith('cross'):
            reference = PROBES / 'evaluation/cross' / task.replace(':', '_')
            with np.load(reference / 'truth.npz') as b:
                for field in ['indices', 'labels', 'distance', 'xy']: np.testing.assert_array_equal(truth[field], b[field])
            for h in heads:
                with np.load(reference / f'{h}__base.npz') as b: cached_base[h] = {k: b[k].copy() for k in b.files}
        mem, mm = context_memory(model, source, context)
        collected = {f'{h}__{kind}': defaultdict(list) for h in heads for kind in ['base', 'memory']}
        errors = {kind: defaultdict(list) for kind in ['base', 'memory']}
        boundary = False; base_replay = True
        for start in range(0, len(ids), 32):
            ii = ids[start:start+32]; features = fs[start:start+32]
            z = torch.from_numpy(source.z(ii)).cuda().float(); act = torch.from_numpy(source.actions[ii]).cuda(); z0 = z[:, 0].contiguous()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                bp = base(z0, act).float()
                pred = model(z0, act, mem.expand(len(ii), -1, -1), mm.expand(len(ii), -1)).float()
            assert torch.equal(pred[:, 0], z0)
            if not context: assert torch.equal(bp, pred)
            for kind, value in [('base', bp), ('memory', pred)]:
                diff = value[:, 1:] - z[:, 1:]
                errors[kind]['latent_l1'].append(diff.abs().mean((1, 2, 3)).cpu().numpy())
                errors[kind]['latent_l2_rms'].append(diff.square().mean((1, 2, 3)).sqrt().cpu().numpy())
                for h, head in heads.items():
                    v = readout(head, value, features, rule)
                    for key, a in v.items(): collected[f'{h}__{kind}'][key].append(a)
                    if not boundary:
                        altered = [f.copy() for f in features]
                        for f in altered: f[1:] = 123
                        check = readout(head, value, altered, rule)
                        for key in v: np.testing.assert_array_equal(v[key], check[key])
            boundary = True
        metrics = {}; raw = {}
        for key, fields in collected.items():
            kind = key.split('__')[1]; h = key.split('__')[0]
            v = {k: np.concatenate(a) for k, a in fields.items()}
            v.update({k: np.concatenate(a) for k, a in errors[kind].items()})
            if kind == 'base' and cached_base:
                for field in v: np.testing.assert_array_equal(v[field], cached_base[h][field])
            raw[key] = v; save_npz(folder / (key + '.npz'), v)
        for key, v in raw.items():
            h, kind = key.split('__')
            metrics[key] = describe(v, truth, labels, mask, rule, green, goals,
                base=raw[h + '__base'] if kind == 'memory' else None)
            b = raw[h + '__base']['log_score']; delta = v['log_score'] - b
            metrics[key]['success_score_change'] = float(delta[mask & labels].mean()) if (mask & labels).any() else None
            metrics[key]['failure_score_change'] = float(delta[mask & ~labels].mean()) if (mask & ~labels).any() else None
        save_npz(folder / 'truth.npz', truth)
        write_json(folder / 'summary.json', {'task': task, 'panel': panel, 'condition': condition, 'context': context,
            'query': query, 'rule': rule, 'green': green, 'dynamic_goals': goals, 'provenance': prov,
            'conditions': metrics, 'future_input_isolation': boundary, 'base_exact_replay': base_replay,
            'base_replay_scope': 'historical_cross' if cached_base else 'recomputed_calibration',
            'first_action_success': row.get('first_action_success'), 'seconds': time.time()-t0})
        print(name, panel, task, condition, round(time.time()-t0), flush=True)
    write_json(out / 'complete.json', {'complete': True, 'provenance': prov, 'specs': len(specs)})


def summarize(root, smoke):
    from attribution_a_metrics import auc, log_score
    m = read(root / 'manifest.json'); prov = provenance(root); seeds = [0] if smoke else m['seeds']
    specs = eval_specs(m, smoke); results = {}; arrays = 0
    for seed in seeds:
        initials = []; common_draws = []; common_outputs = []
        for arm in ARMS:
            name = f'{arm}_s{seed}'; d = root / ('smoke_models' if smoke else 'models') / name
            c = read(d / 'complete.json'); assert c['provenance'] == prov and c['frozen_modules_unchanged']
            assert digest(d / 'memory.pt') == c['checkpoint_sha256']
            initials.append(c['checks']['initial_model_sha256']); common_draws.append(c['common_draw_sha256']); common_outputs.append(c['checks']['common_input_output_sha256'])
            draws = read(d / 'draws.json'); rr = [r for r in m['rows'] if r['split'] == 'train' and (not smoke or r['task'] in m['smoke_train_tasks'])]
            for step, draw in enumerate(draws):
                row, ids, times, ctx, category = episode(rr, seed, step, arm)
                assert (draw['task'], draw['query'], draw['times'], draw['context'], draw['category']) == (row['task'], ids, times, ctx, category)
            folder = root / ('smoke_eval' if smoke else 'evaluation') / name
            ec = read(folder / 'complete.json'); assert ec['complete'] and ec['specs'] == len(specs)
            assert ec['provenance']['manifest'] == prov['manifest'] and ec['provenance']['code'] == prov['code']
            assert ec['provenance']['checkpoint'] == c['checkpoint_sha256']
            rows = []
            for panel, spec, condition, context in specs:
                path = folder / panel / spec['task'].replace(':', '_') / condition
                r = read(path / 'summary.json'); assert r['provenance'] == ec['provenance']
                assert r['context'] == context and r['future_input_isolation'] and r['base_exact_replay']
                with np.load(path / 'truth.npz') as t: truth = {k: t[k] for k in t.files}
                mask = truth['query_mask']; y = truth['labels']; assert truth['indices'][mask].tolist() == r['query']
                for key, metrics in r['conditions'].items():
                    with np.load(path / (key + '.npz')) as v:
                        np.testing.assert_allclose(log_score(v['distance'], r['rule']), v['log_score'], rtol=1e-12, atol=1e-12)
                        assert auc(y[mask], v['log_score'][mask]) == metrics['auc']
                        computed = describe(v, truth, y, mask, r['rule'], r['green'], r['dynamic_goals'])
                        for field in ['auc', 'first_success_rank', 'hit_at_10', 'all', 'success', 'failure']: assert computed[field] == metrics[field]
                        h, kind = key.split('__'); reference = r['conditions']['original__' + kind]
                        assert metrics['all']['latent_l1'] == reference['all']['latent_l1']
                        # Independently verify score shifts and pair flips with base arrays.
                        with np.load(path / (h + '__base.npz')) as b:
                            delta = v['log_score'] - b['log_score']
                            for label, use in [('success', mask & y), ('failure', mask & ~y)]:
                                expected = float(delta[use].mean()) if use.any() else None
                                assert metrics[label + '_score_change'] == expected
                        arrays += 1
                rows.append(r)
            results[name] = rows
        assert len(set(initials)) == len(set(common_draws)) == len(set(common_outputs)) == 1
    if smoke:
        write_json(root / 'smoke_check.json', {'passed': True, 'provenance': prov, 'arrays_verified': arrays,
            'paired_training': True, 'scope': 'Engineering checks only; no performance gate'})
        print('SMOKE_PASSED', arrays, flush=True); return
    primary = m['primary_head'] + '__memory'; baseref = m['primary_head'] + '__base'
    summary = {'complete': True, 'audit_passed': True, 'provenance': prov, 'arrays_verified': arrays,
        'scope': m['scope'], 'coverage': m['coverage'], 'macros': {}, 'paired': {}}
    groups = sorted({(panel, condition) for panel, _, condition, _ in specs})
    for panel, condition in groups:
        tag = panel + '/' + condition; summary['macros'][tag] = {}
        for name, rows in results.items():
            selected = [r for r in rows if r['panel'] == panel and r['condition'] == condition and (panel != 'cross_natural' or r['context'])]
            values = [r['conditions'][primary]['auc'] for r in selected if r['conditions'][primary]['auc'] is not None]
            bases = [r['conditions'][baseref]['auc'] for r in selected if r['conditions'][baseref]['auc'] is not None]
            summary['macros'][tag][name] = {'scenes': len(selected), 'auc_scenes': len(values),
                'auc': float(np.mean(values)) if values else None, 'base_auc': float(np.mean(bases)) if bases else None,
                'per_task': [{'task': r['task'], 'auc': r['conditions'][primary]['auc'], 'base_auc': r['conditions'][baseref]['auc']} for r in selected]}
        summary['paired'][tag] = {}
        for a, b in [('near', 'uniform'), ('mixed', 'uniform'), ('mixed', 'near')]:
            by = defaultdict(list); seedmeans = {}
            for seed in seeds:
                aa = summary['macros'][tag][f'{a}_s{seed}']['per_task']; bb = summary['macros'][tag][f'{b}_s{seed}']['per_task']
                assert [r['task'] for r in aa] == [r['task'] for r in bb]
                diffs = []
                for x, y in zip(aa, bb):
                    assert x['base_auc'] == y['base_auc']
                    if x['auc'] is not None:
                        value = x['auc'] - y['auc']; by[x['task']].append(value); diffs.append(value)
                seedmeans[str(seed)] = float(np.mean(diffs)) if diffs else None
            tasks = sorted(by); delta = np.array([np.mean(by[t]) for t in tasks]); rng = np.random.default_rng(SEED)
            samples = np.zeros(10000)
            for tp in sorted({t[:5] for t in tasks}):
                vals = delta[[i for i, t in enumerate(tasks) if t[:5] == tp]]
                samples += rng.choice(vals, (10000, len(vals)), replace=True).sum(1)
            ci = (np.quantile(samples / len(tasks), [.025, .975]).tolist() if tasks else None)
            summary['paired'][tag][a + '-' + b] = {'n_scenes': len(tasks), 'mean': float(delta.mean()) if tasks else None,
                'scene_bootstrap_ci95_conditional_on_templates_and_trained_models': ci, 'per_seed': seedmeans,
                'by_template': {tp: float(np.mean([np.mean(by[t]) for t in tasks if t[:5] == tp])) for tp in sorted({t[:5] for t in tasks})}}
    write_json(root / 'SUMMARY.json', summary)
    lines = ['# Support sampling pilot results', '', m['scope'], '', '| Model | Natural first failure AUC | Same head base AUC | Scenes |', '|---|---:|---:|---:|']
    for name, v in summary['macros']['cross_natural/natural'].items():
        lines.append(f'| {name} | {v["auc"]} | {v["base_auc"]} | {v["auc_scenes"]} |')
    lines += ['', 'All seeds and three sampling arms retained. Selected-support panels, denominators and paired intervals are in SUMMARY.json.',
        'These are reused-D0 fixed-query results, not new-scene confirmation or closed-loop success rates. No learned-selection claim from K1 alone.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n'); print('SUMMARY_COMPLETE', arrays, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['inventory', 'score', 'finalize', 'train', 'eval', 'summary'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1); p.add_argument('--arm', choices=ARMS, default='uniform')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); torch.set_num_threads(4)
    if a.stage == 'inventory': inventory(a.root)
    elif a.stage == 'score': score_support(a.root, a.shard, a.nshards)
    elif a.stage == 'finalize': finalize(a.root)
    elif a.stage == 'train': train(a.root, a.arm, a.seed, a.smoke)
    elif a.stage == 'eval': evaluate(a.root, a.arm, a.seed, a.smoke)
    else: summarize(a.root, a.smoke)


if __name__ == '__main__': main()
