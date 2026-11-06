import argparse, json, os
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from common import Predictor, AttnProbe, MemoryPredictor, surface_dist, rule_score

p = argparse.ArgumentParser()
p.add_argument('--packs', default='/mnt/meg/jjenchen/phyre-unc/packs')
p.add_argument('--dev', default='within_f0_dev_clusters_pca128')
p.add_argument('--predictor', required=True)
p.add_argument('--probe', default='/path/to/phyre-unc/results/probes_pca128/ball_within_template_f0_full.pt')
p.add_argument('--episodes', default='/path/to/phyre-unc/results/tier1_episodes_within_f0_dev.json')
p.add_argument('--out', required=True)
p.add_argument('--memory', default='')
p.add_argument('--empty-memory', action='store_true')
args = p.parse_args()
dev = torch.device('cuda')
W, G = 256, 14
GREEN, PURPLE, BLACK = 9, 11, 13


def make_episodes(tasks, clusters, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(lambda: defaultdict(list))
    for i, (t, c) in enumerate(zip(tasks, clusters)):
        by[t][int(c)].append(i)
    eps = []
    for t in sorted(by):
        cl = by[t]
        for c in sorted(cl):
            ctx = [int(rng.choice(cl[c]))]
            eps.append({'task': t, 'kind': 'k1', 'context': ctx, 'context_clusters': [c]})
        for k in (2, 4):
            cs = sorted(rng.choice(sorted(cl), min(k, len(cl)), replace=False).tolist())
            eps.append({'task': t, 'kind': f'k{k}', 'context': [int(rng.choice(cl[c])) for c in cs], 'context_clusters': cs})
    for e in eps:
        cl = by[e['task']]
        e['queries'] = [i for c in cl for i in cl[c] if i not in e['context']]
        e['near'] = [i for i in e['queries'] if int(clusters[i]) in e['context_clusters']]
    return eps


def object_token_mask(f):
    T, N, _ = f.shape
    m = np.zeros((T, G, G), bool)
    for o in range(N):
        if f[0, o, BLACK] > .5 or f[0, o, PURPLE] > .5 or f[0, o, :].sum() == 0:
            continue
        r = f[0, o, 3] * G / 2 + 1
        for t in range(T):
            cx, cy = f[t, o, 0] * G, (1 - f[t, o, 1]) * G
            ys, xs = np.mgrid[0:G, 0:G]
            m[t] |= (xs + .5 - cx) ** 2 + (ys + .5 - cy) ** 2 <= r * r
    return m.reshape(T, G * G)


lat = np.load(os.path.join(args.packs, args.dev + '_latents.npy'), mmap_mode='r')
meta = np.load(os.path.join(args.packs, args.dev + '_meta.npz'))
F_, M_, A_, Y_, T_ = meta['features'], meta['mask'], meta['action'], meta['solved'], meta['tasks']
rows = [json.loads(l) for l in open('/path/to/phyre-unc/data/manifests/ball_within_template_f0_dev_clusters.jsonl')]
file2cluster = {r['file']: r['cluster'] for r in rows}
C_ = np.array([file2cluster[str(f)] for f in meta['files']])
if os.path.exists(args.episodes):
    eps = json.load(open(args.episodes))
else:
    eps = make_episodes(T_, C_)
    json.dump(eps, open(args.episodes, 'w'))

pc = torch.load(args.predictor, map_location='cpu', weights_only=False); pa = pc['args']
pred = Predictor(pc['Din'], pc['T'], pc['S'], pa['d'], pa['layers'], pa['heads']).to(dev)
pred.load_state_dict(pc['state_dict']); pred.eval()
qc = torch.load(args.probe, map_location='cpu', weights_only=False)
probe = AttnProbe(qc['D'], qc['S']).to(dev); probe.load_state_dict(qc['state_dict']); probe.eval()
rule = qc['rule']

N = len(lat)
T_full = lat.shape[1]


def metrics(pr, z, idx):
    err = (pr - z).abs().mean(-1).cpu().numpy()
    B, T = err.shape[:2]
    kv = probe.inp(pr.reshape(B * T, G * G, -1)) + probe.space
    out = []
    for b in range(B):
        k = idx[b]
        f = F_[k][:, M_[k]]
        om = object_token_mask(f)
        dyn = [o for o in range(f.shape[1]) if f[0, o, BLACK] < .5 and f[0, o, PURPLE] < .5]
        pf = np.repeat(f[:1], T, 0).copy()
        for o in dyn:
            q = torch.from_numpy(np.repeat(f[:1, o], T, 0)).to(dev)
            pxy, pang = probe.read(kv[b * T:(b + 1) * T], q)
            pf[:, o, :2] = pxy.cpu().numpy() / W
            pang = pang.cpu().numpy()
            pf[:, o, 2] = (np.arctan2(pang[:, 1], pang[:, 0]) / 2) / (2 * np.pi) % 1
        g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
        out.append((err[b, 1:].mean(), err[b, 1:][om[1:]].mean() if om[1:].any() else np.nan,
                    np.median(np.linalg.norm((pf[1:, g, :2] - f[1:, g, :2]) * W, axis=-1)), rule_score(surface_dist(pf), rule)))
    return out


def load_z(idx):
    return torch.from_numpy(np.stack([lat[i] for i in idx])).to(dev).float()


with torch.no_grad():
    base_metrics = np.zeros((N, 4))
    for i in range(0, N, 16):
        idx = list(range(i, min(i + 16, N)))
        z = load_z(idx)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pr = pred(z[:, 0], torch.from_numpy(A_[idx]).to(dev)).float()
        base_metrics[idx] = metrics(pr, z, idx)
lat_err, obj_err, pos_err, score = base_metrics.T.copy()
mem_model = None
if args.memory:
    mc = torch.load(args.memory, map_location='cpu', weights_only=False)
    mem_model = MemoryPredictor(pred, pc['Din'], mode=mc['mode'], mem_steps=mc.get('mem_steps', 8), mem_grid=mc.get('mem_grid', 7), per_attempt=mc.get('per_attempt', False), gate_empty=mc.get('gate_empty', False)).to(dev)
    mem_model.load_state_dict(mc['state_dict'], strict=False)
    mem_model.eval()


def episode_metrics(e):
    if mem_model is None:
        return {q: base_metrics[q] for q in e['queries']}
    ctx = [] if args.empty_memory else e['context']
    K = max(len(ctx), 1)
    full = mem_model.mode == 'residual'
    ms = mem_model.mem_steps
    c_lat = torch.zeros((1, K, T_full if full else ms) + lat.shape[2:], device=dev)
    c_act = torch.zeros(1, K, 3, device=dev); c_out = torch.zeros(1, K, device=dev); c_m = torch.zeros(1, K, dtype=torch.bool, device=dev)
    for j, ci in enumerate(ctx):
        c_lat[0, j] = torch.from_numpy(np.ascontiguousarray(lat[ci] if full else lat[ci, ::T_full // ms])).to(dev).float()
        c_act[0, j] = torch.from_numpy(A_[ci]).to(dev); c_out[0, j] = float(Y_[ci]); c_m[0, j] = True
    res_e = {}
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            res_c = mem_model.context_residual(c_lat, c_act).float() if full else None
            mem, mm = mem_model.encode_memory(c_lat, c_act, c_out, c_m, res_c)
        for i in range(0, len(e['queries']), 16):
            idx = e['queries'][i:i + 16]
            z = load_z(idx)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pr = mem_model(z[:, 0], torch.from_numpy(A_[idx]).to(dev), mem.expand(len(idx), -1, -1), mm.expand(len(idx), -1)).float()
            for q, mtr in zip(idx, metrics(pr, z, idx)):
                res_e[q] = np.array(mtr)
    return res_e


if not any(e['kind'] == 'scorer_fail' for e in eps):
    for t in sorted(set(T_)):
        idx = np.flatnonzero(T_ == t)
        fails = idx[~Y_[idx].astype(bool)]
        if len(fails) == 0:
            continue
        c = int(fails[np.argmax(score[fails])])
        q = [int(i) for i in idx if i != c]
        eps.append({'task': str(t), 'kind': 'scorer_fail', 'context': [c], 'context_clusters': [int(C_[c])],
                    'queries': q, 'near': [i for i in q if C_[i] == C_[c]], 'selected_by': args.predictor})
    json.dump(eps, open(args.episodes, 'w'))


EPM = {}
res = {'predictor': args.predictor, 'memory': args.memory, 'empty_memory': args.empty_memory, 'n_rollouts': N, 'n_episodes': len(eps)}
for kind in ['k1', 'k2', 'k4', 'scorer_fail']:
    E = [(ei, e) for ei, e in enumerate(eps) if e['kind'] == kind]
    for ei, e in E:
        EPM[ei] = episode_metrics(e)
    out = {}
    for split in ['all', 'near', 'far']:
        pick = lambda e: e['queries'] if split == 'all' else (e['near'] if split == 'near' else [i for i in e['queries'] if i not in e['near']])
        rows_ = [(ei, q) for ei, e in E for q in pick(e)]
        if not rows_:
            out[split] = {'n_queries': 0}
            continue
        M4 = np.array([EPM[ei][q] for ei, q in rows_])
        aucs = []
        for ei, e in E:
            qs = pick(e); y = Y_[qs]
            if 0 < y.sum() < len(y):
                aucs.append(roc_auc_score(y, [EPM[ei][q][3] for q in qs]))
        out[split] = {'n_queries': len(rows_), 'latent_l1': round(float(M4[:, 0].mean()), 5),
                      'object_latent_l1': round(float(np.nanmean(M4[:, 1])), 5),
                      'green_err_median_px': round(float(np.median(M4[:, 2])), 2),
                      'rank_auc_mean': round(float(np.mean(aucs)), 4) if aucs else None}
    res[kind] = out
res['per_episode_query'] = {str(ei): {str(q): [float(x) for x in v] for q, v in d.items()} for ei, d in EPM.items()}
json.dump(res, open(args.out, 'w'))
print(json.dumps({k: v for k, v in res.items() if k != 'per_episode_query'}, indent=1))
