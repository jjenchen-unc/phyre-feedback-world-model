"""Fixed-support diagnostic; uses saved outcomes only for support and scoring.

No training. Visual-zero controls are interventions on the existing residual
checkpoint, not separately trained action/outcome-only baselines.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import Predictor, MemoryPredictor, AttnProbe, surface_dist, rule_score


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--root', default='/mnt/meg/jjenchen/phyre-unc')
    p.add_argument('--project', default='/path/to/phyre-unc')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    out = Path(args.out) / args.task.replace(':', '_')
    out.mkdir(parents=True, exist_ok=False)
    root, project = Path(args.root), Path(args.project)
    pack = root / 'packs/cross_f1_dev_shortlist1024_pca128'
    lat = np.load(str(pack) + '_latents.npy', mmap_mode='r')
    meta = np.load(str(pack) + '_meta.npz')
    features, masks, actions, labels, tasks, files = (
        meta[k] for k in ['features', 'mask', 'action', 'solved', 'tasks', 'files'])
    labels = labels.astype(bool)
    idx = np.flatnonzero(tasks == args.task)
    assert len(idx) == 1024
    paths = {
        'base': root / 'runs/cross_f1_base_s16r16_pca128/predictor.pt',
        'original': root / 'runs/mem_cross_f1_residual_hires_clusters/memory.pt',
        'gated': root / 'runs/mem_cross_f1_residual_hires_clusters_gate/memory.pt',
        'probe': project / 'results/probes_pca128_cross/ball_cross_template_f1_full.pt',
    }
    pc = torch.load(paths['base'], map_location='cpu', weights_only=False)
    pa = pc['args']
    dev = torch.device('cuda')
    base = Predictor(pc['Din'], pc['T'], pc['S'], pa['d'], pa['layers'], pa['heads']).to(dev)
    base.load_state_dict(pc['state_dict']); base.eval()
    models, checkpoints = {}, {}
    for key in ['original', 'gated']:
        mc = torch.load(paths[key], map_location='cpu', weights_only=False)
        assert mc['base'] == str(paths['base']) and not mc.get('unfreeze', False)
        model = MemoryPredictor(base, pc['Din'], mode=mc['mode'], mem_steps=mc['mem_steps'],
                                mem_grid=mc['mem_grid'], per_attempt=mc['per_attempt'],
                                gate_empty=mc.get('gate_empty', False)).to(dev)
        missing, unexpected = model.load_state_dict(mc['state_dict'], strict=False)
        assert not unexpected and all(k.startswith('base.') for k in missing)
        models[key] = model.eval()
        checkpoints[key] = {k: v for k, v in mc.items() if k not in ['state_dict', 'log']}
    qc = torch.load(paths['probe'], map_location='cpu', weights_only=False)
    probe = AttnProbe(qc['D'], qc['S']).to(dev)
    probe.load_state_dict(qc['state_dict']); probe.eval()
    rule = qc['rule']
    t0 = time.time()

    @torch.no_grad()
    def encode(model, context, control):
        k = max(len(context), 1)
        z = torch.zeros((1, k) + lat.shape[1:], device=dev)
        a = torch.zeros(1, k, 3, device=dev)
        y = torch.zeros(1, k, device=dev)
        mask = torch.zeros(1, k, dtype=torch.bool, device=dev)
        for j, i in enumerate(context):
            z[0, j] = torch.from_numpy(np.array(lat[i])).to(dev).float()
            a[0, j] = torch.from_numpy(actions[i]).to(dev)
            y[0, j] = float(labels[i]); mask[0, j] = True
        with torch.autocast('cuda', dtype=torch.bfloat16):
            residual = model.context_residual(z, a).float()
            if control in ['no_residual', 'action_outcome_only']:
                residual.zero_()
            if control in ['no_video', 'action_outcome_only']:
                z.zero_()
            return model.encode_memory(z, a, y, mask, residual)

    @torch.no_grad()
    def evaluate(name, model=None, context=(), scale=1., control='full', observed=False):
        target = out / (name + '.npz')
        if target.exists():
            return dict(np.load(target))
        if model is not None:
            model.out_scale = scale
            memory, memory_mask = encode(model, context, control)
        data = {k: [] for k in ['scores', 'latent_l1_time', 'green_xy', 'green_error',
                                'surface_distance', 'surface_error']}
        for s in range(0, len(idx), 32):
            ids = idx[s:s + 32]
            truth = torch.from_numpy(np.stack([lat[i] for i in ids])).to(dev).float()
            a = torch.from_numpy(actions[ids]).to(dev)
            if observed:
                pr = truth
            else:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pr = (base(truth[:, 0], a) if model is None else model(
                        truth[:, 0], a, memory.expand(len(ids), -1, -1),
                        memory_mask.expand(len(ids), -1))).float()
            data['latent_l1_time'].extend((pr - truth).abs().mean((2, 3)).cpu().numpy())
            b, nt = pr.shape[:2]
            kv = probe.inp(pr.reshape(b * nt, 196, -1)) + probe.space
            for j, i in enumerate(ids):
                f = features[i][:, masks[i]]
                pf = np.repeat(f[:1], nt, axis=0).copy()
                dyn = [o for o in range(f.shape[1]) if f[0, o, 13] < .5 and f[0, o, 11] < .5]
                for o in dyn:
                    q = torch.from_numpy(np.repeat(f[:1, o], nt, axis=0)).to(dev)
                    xy, angle = probe.read(kv[j * nt:(j + 1) * nt], q)
                    pf[:, o, :2] = xy.cpu().numpy() / 256
                    angle = angle.cpu().numpy()
                    pf[:, o, 2] = np.arctan2(angle[:, 1], angle[:, 0]) / (4 * np.pi) % 1
                green = int(np.flatnonzero(f[0, :, 9] > .5)[0])
                dist, real_dist = surface_dist(pf), surface_dist(f)
                data['scores'].append(rule_score(dist, rule))
                data['green_xy'].append(pf[:, green, :2] * 256)
                data['green_error'].append(np.linalg.norm((pf[:, green, :2] - f[:, green, :2]) * 256, axis=-1))
                data['surface_distance'].append(dist)
                data['surface_error'].append(np.abs(dist - real_dist))
        data = {k: np.asarray(v) for k, v in data.items()}
        np.savez_compressed(target, **data)
        print(json.dumps({'task': args.task, 'condition': name, 'seconds': round(time.time() - t0)}), flush=True)
        return data

    def describe(ids):
        return [{'pack_index': int(i), 'task': str(tasks[i]), 'file': str(files[i]),
                 'action': actions[i].tolist(), 'solved': bool(labels[i])} for i in ids]

    base_data = evaluate('base')
    order = np.argsort(-base_data['scores'], kind='stable')
    first = int(np.flatnonzero(labels[idx][order])[0]) + 1
    reference = json.load(open(project / 'results/cx_clm_base.json'))
    historical = next(r['first_success'] for r in reference['per_task'] if r['task'] == args.task)
    assert first == historical, (first, historical)
    # These are the genuinely observed failed prefix of the frozen base policy.
    support = idx[order[:min(4, first - 1)]]
    assert len(support) >= 1 and not labels[support].any()
    true_scores = np.array([rule_score(surface_dist(features[i][:, masks[i]]), rule) for i in idx])
    green_truth = np.stack([features[i][:, masks[i]][:, np.flatnonzero(features[i][0, masks[i], 9] > .5)[0], :2] * 256 for i in idx])
    np.savez_compressed(out / 'candidates.npz', indices=idx, actions=actions[idx], labels=labels[idx],
                        files=files[idx], true_scores=true_scores, green_truth=green_truth)
    observed = evaluate('observed_latents', observed=True)
    empty = evaluate('empty_original', models['original'])
    empty_gate = evaluate('empty_gated', models['gated'])
    assert np.allclose(base_data['scores'], empty_gate['scores'], atol=1e-5, rtol=1e-5)

    def summarize(data, excluded):
        query = ~np.isin(idx, excluded)
        positions = np.flatnonzero(query)
        ranked = positions[np.argsort(-data['scores'][query], kind='stable')]
        ranks = np.empty(len(idx), int); ranks.fill(-1); ranks[ranked] = np.arange(1, len(ranked) + 1)
        pos = query & labels[idx]
        top_false = ranked[:10][~labels[idx][ranked[:10]]]
        def metrics(sel):
            return {'n': int(sel.sum()),
                    'latent_l1': float(data['latent_l1_time'][sel, 1:].mean()),
                    'green_error_median_px': float(np.median(data['green_error'][sel, 1:])),
                    'surface_error_mean_px': float(data['surface_error'][sel, 1:].mean()),
                    'score_mean': float(data['scores'][sel].mean())}
        return {'first_success_rank': int(np.flatnonzero(labels[idx][ranked])[0]) + 1,
                'rank_auc': float(roc_auc_score(labels[idx][query], data['scores'][query])),
                'all': metrics(query), 'successful_candidates': metrics(pos),
                'success_ranks': ranks[pos].tolist(),
                'top10': describe(idx[ranked[:10]]),
                'top10_false_positive_mean_true_score': float(true_scores[top_false].mean()) if len(top_false) else None}

    report = {'task': args.task, 'historical_base_first_success': historical,
              'base_reproduced': first, 'support_selection': 'failed prefix of frozen base ranking; no successful support',
              'support': describe(support), 'rule': rule,
              'checkpoint_metadata': checkpoints, 'paths': {k: str(v) for k, v in paths.items()},
              'sha256': {k: sha(v) for k, v in paths.items()},
              'script_sha256': sha(__file__), 'common_sha256': sha(Path(__file__).with_name('common.py')),
              'controls_warning': 'Visual-zero ablations are OOD interventions, not trained action-only baselines. Wrong-scene controls replace action/video/outcome coherently.',
              'contexts': {}}
    for k in sorted({1, len(support)}):
        ctx = support[:k].tolist()
        conditions = {'base': base_data, 'observed_latents': observed,
                      'empty_original': empty, 'empty_gated': empty_gate}
        donors = {}
        for donor_task, label in [('00014:009', 'wrong_scene_same_template'),
                                   ('00000:000', 'wrong_scene_other_template')]:
            donor = np.flatnonzero((tasks == donor_task) & ~labels)[:k].tolist()
            assert len(donor) == k and not set(donor) & set(idx)
            donors[label] = donor
        settings = [('original_full', 'original', ctx, 1., 'full'),
                    ('gated_full', 'gated', ctx, 1., 'full'),
                    ('gated_scale025', 'gated', ctx, .25, 'full'),
                    ('action_outcome_only', 'gated', ctx, 1., 'action_outcome_only'),
                    ('no_residual', 'gated', ctx, 1., 'no_residual'),
                    ('no_video', 'gated', ctx, 1., 'no_video')]
        settings += [(label, 'gated', donor, 1., 'full') for label, donor in donors.items()]
        for name, model_key, context, scale, control in settings:
            conditions[name] = evaluate(f'k{k}_{name}', models[model_key], context, scale, control)
        report['contexts'][str(k)] = {'support': describe(ctx), 'donors': {n: describe(v) for n, v in donors.items()},
                                    'conditions': {name: summarize(d, ctx) for name, d in conditions.items()}}
        report['elapsed_seconds'] = round(time.time() - t0)
        tmp = out / 'summary.tmp.json'
        tmp.write_text(json.dumps(report, indent=2)); os.replace(tmp, out / 'summary.json')
        print(json.dumps({'task': args.task, 'K': k, 'ranks': {n: d['first_success_rank'] for n, d in report['contexts'][str(k)]['conditions'].items()}}), flush=True)
    print('DONE', args.task, flush=True)


if __name__ == '__main__':
    main()
