"""Same-query latent/geometry/ranking measurement with common frozen base contexts."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from common import MemoryPredictor, surface_dist, rule_score
from redesign_lib import load_models, digest, write_json, PROJECT
from attribution_a_metrics import log_score, describe


def save_arrays(path, data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **data)
    tmp.replace(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((a.root / 'manifest.json').read_text())
    assert digest(manifest['pack'] + '_meta.npz') == manifest['meta_sha256']
    assert digest(manifest['base']) == manifest['base_sha256'] and digest(manifest['probe']) == manifest['probe_sha256']
    lat = np.load(manifest['pack'] + '_latents.npy', mmap_mode='r')
    with np.load(manifest['pack'] + '_meta.npz') as packed: meta = {k: packed[k] for k in packed.files}
    base, probe, rule, pc, _ = load_models(); assert rule[0] == 'soft'
    models = {}
    for name, spec in manifest['models'].items():
        assert digest(spec['path']) == spec['sha256']
        mc = torch.load(spec['path'], map_location='cpu', weights_only=False)
        assert mc['base'] == manifest['base'] and not mc.get('unfreeze', False)
        model = MemoryPredictor(base, pc['Din'], mode=mc['mode'], mem_steps=mc['mem_steps'], mem_grid=mc['mem_grid'], per_attempt=mc['per_attempt'], gate_empty=True).cuda()
        missing, unexpected = model.load_state_dict(mc['state_dict'], strict=False)
        assert not unexpected and all(k.startswith('base.') for k in missing)
        models[name] = model.eval()
    code_hashes = {f.name: digest(f) for f in Path(__file__).parent.glob('*.py')}
    provenance = {'manifest': digest(a.root / 'manifest.json'), 'code': code_hashes}
    selected = [r for r in manifest['tasks'] if not a.smoke or r['task'] in manifest['smoke_tasks']]
    selected = selected[a.shard::a.nshards]
    out = a.root / ('smoke' if a.smoke else 'tasks'); out.mkdir(exist_ok=True, parents=True)

    def encode(model, ctx):
        k = max(1, len(ctx)); z = torch.zeros((1, k) + lat.shape[1:], device='cuda')
        acts = torch.zeros(1, k, 3, device='cuda'); y = torch.zeros(1, k, device='cuda')
        mask = torch.zeros(1, k, dtype=torch.bool, device='cuda')
        for j, i in enumerate(ctx):
            assert not meta['solved'][i]
            z[0, j] = torch.from_numpy(np.array(lat[i])).cuda().float()
            acts[0, j] = torch.from_numpy(meta['action'][i]).cuda(); mask[0, j] = True
        with torch.autocast('cuda', dtype=torch.bfloat16):
            residual = model.context_residual(z, acts).float()
            return model.encode_memory(z, acts, y, mask, residual)

    def readout(pred, features):
        b, nt, ns, _ = pred.shape
        kv = probe.inp(pred.float().reshape(b * nt, ns, -1)) + probe.space
        ds, xys, old = [], [], []
        for j, f in enumerate(features):
            pf = np.repeat(f[:1], nt, 0).copy()
            for o in range(f.shape[1]):
                if f[0, o, 13] >= .5 or f[0, o, 11] >= .5: continue
                q = torch.from_numpy(np.repeat(f[:1, o], nt, 0)).cuda()
                xy, angle = probe.read(kv[j * nt:(j + 1) * nt], q)
                pf[:, o, :2] = xy.cpu().numpy() / 256
                angle = angle.cpu().numpy()
                pf[:, o, 2] = (np.arctan2(angle[:, 1], angle[:, 0]) / 2) / (2 * np.pi) % 1
            d = surface_dist(pf); ds.append(d); xys.append(pf[:, :, :2] * 256); old.append(rule_score(d, rule))
        d = np.stack(ds)
        return {'distance': d, 'xy': np.stack(xys), 'legacy_score': np.array(old), 'log_score': log_score(d, rule)}

    def evaluate(ids, features, condition, ctx=(), model=None):
        mem = encode(model, ctx) if model is not None else None
        collected = {}
        for start in range(0, len(ids), 32):
            ii = ids[start:start + 32]
            truth = torch.from_numpy(np.stack([lat[i] for i in ii])).cuda().float()
            act = torch.from_numpy(meta['action'][ii]).cuda(); z0 = truth[:, 0].contiguous()
            if condition == 'observed_latent': pred = truth
            else:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = base(z0, act) if model is None else model(z0, act, mem[0].expand(len(ii), -1, -1), mem[1].expand(len(ii), -1))
                pred = pred.float()
            assert torch.equal(pred[:, 0], truth[:, 0])
            values = readout(pred, features[start:start + 32])
            diff = pred[:, 1:] - truth[:, 1:]
            values['latent_l1'] = diff.abs().mean((1, 2, 3)).cpu().numpy()
            values['latent_l2_rms'] = diff.square().mean((1, 2, 3)).sqrt().cpu().numpy()
            values['latent_l1_time'] = diff.abs().mean((2, 3)).cpu().numpy()
            for k, v in values.items(): collected.setdefault(k, []).append(v)
        return {k: np.concatenate(v) for k, v in collected.items()}

    start_time = time.time()
    for entry in selected:
        task = entry['task']; stem = task.replace(':', '_'); dest = out / stem; dest.mkdir(exist_ok=True)
        done = dest / 'summary.json'
        if done.exists():
            assert json.loads(done.read_text())['provenance'] == provenance
            continue
        ids = np.array(entry['indices']); labels = meta['solved'][ids].astype(bool)
        mask = np.isin(ids, entry['query']); assert mask.sum() == len(entry['query'])
        features = [meta['features'][i][:, meta['mask'][i]] for i in ids]
        assert len({f.shape for f in features}) == 1
        green = int(np.flatnonzero(features[0][0, :, 9] > .5)[0])
        goals = [o for o in range(features[0].shape[1]) if features[0][0, o, 10] > .5 and features[0][0, o, 13] < .5 and features[0][0, o, 11] < .5]
        true_d = np.stack([surface_dist(f) for f in features])
        truth = {'distance': true_d, 'xy': np.stack([f[:, :, :2] * 256 for f in features]),
                 'log_score': log_score(true_d, rule), 'latent_l1': np.full(len(ids), np.nan), 'latent_l2_rms': np.full(len(ids), np.nan)}
        dense = []
        for i, f in zip(ids, features):
            with np.load(PROJECT / 'data' / str(meta['files'][i])) as raw:
                assert int(raw['stride']) == 8 and int(raw['steps']) == 1017
                ff = raw['features']; assert len(ff) == 128
                assert bool(int(raw['status']) > 0) == bool(meta['solved'][i])
                np.testing.assert_array_equal(raw['action'], meta['action'][i])
                np.testing.assert_array_equal(ff[::4], f)
                dense.append(surface_dist(ff[:125]))  # Same endpoints: simulator step0..992.
        truth['dense_distance'] = np.stack(dense)
        truth['coarse_weighted_log_score'] = log_score(true_d, rule, weighted=True)
        truth['dense_weighted_log_score'] = log_score(truth['dense_distance'], rule, weighted=True)
        truth['indices'] = ids; truth['labels'] = labels; truth['query_mask'] = mask
        truth['action'] = meta['action'][ids]; truth['files'] = meta['files'][ids]
        save_arrays(dest / 'truth.npz', truth)
        base_data = evaluate(ids, features, 'base')
        refpath = Path(manifest['base_trace_source']) / 'base_stable_log.json.tasks' / (stem + '_scores.npz')
        with np.load(refpath) as ref:
            np.testing.assert_array_equal(ids, ref['indices'])
            np.testing.assert_array_equal(base_data['legacy_score'], ref['scores'][0, 0])
            np.testing.assert_array_equal(base_data['log_score'], ref['scores'][0, 1])
        save_arrays(dest / 'base.npz', base_data)
        obs = evaluate(ids, features, 'observed_latent'); save_arrays(dest / 'observed_latent.npz', obs)
        metrics = {'base': describe(base_data, truth, labels, mask, rule, green, goals),
                   'observed_latent': describe(obs, truth, labels, mask, rule, green, goals),
                   'true_geometry': describe(truth, truth, labels, mask, rule, green, goals)}
        for resolution in ['coarse', 'dense']:
            proxy = dict(truth); proxy['log_score'] = truth[resolution + '_weighted_log_score']
            metrics['true_geometry_' + resolution + '_weighted'] = describe(proxy, truth, labels, mask, rule, green, goals)
        names = ['initial_R0'] if a.smoke else list(models)
        # Exercise every checkpoint's hard-empty path on a full batch, even in smoke.
        empty_parity = []
        for name, model in models.items():
            empty = evaluate(ids[:32], features[:32], 'memory', [], model)
            for field in ['log_score', 'legacy_score', 'latent_l1', 'distance', 'xy']:
                np.testing.assert_array_equal(empty[field], base_data[field][:32])
            empty_parity.append(name)
        for name in names:
            for level, ctx in entry['contexts'].items():
                condition = name + '_' + level
                if ctx:
                    value = evaluate(ids, features, 'memory', ctx, models[name])
                    if level == 'K1':
                        refpath = Path(manifest['base_trace_source']) / (name + '_stable_log.json.tasks') / (stem + '_scores.npz')
                        with np.load(refpath) as ref:
                            np.testing.assert_array_equal(value['legacy_score'], ref['scores'][1, 0])
                            np.testing.assert_array_equal(value['log_score'], ref['scores'][1, 1])
                    save_arrays(dest / (condition + '.npz'), value)
                else: value = base_data
                metrics[condition] = describe(value, truth, labels, mask, rule, green, goals, base=base_data)
        record = {'task': task, 'provenance': provenance, 'entry': entry, 'rule': rule, 'green': green, 'dynamic_goals': goals,
                  'conditions': metrics, 'base_score_exact_replay': True, 'memory_K1_score_exact_replay': True,
                  'empty_parity_models': empty_parity, 'dense_raw_checked': len(ids), 'seconds_cumulative': round(time.time() - start_time)}
        write_json(done, record)
        print(json.dumps({'task': task, 'conditions': len(metrics), 'seconds': round(time.time() - start_time)}), flush=True)
    write_json(out / f'complete_{a.shard}.json', {'complete': True, 'tasks': [r['task'] for r in selected], 'provenance': provenance})


if __name__ == '__main__': main()
