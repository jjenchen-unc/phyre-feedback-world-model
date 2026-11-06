import argparse, glob, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument('--latents', required=True)
p.add_argument('--data', default='/path/to/phyre-unc/data')
p.add_argument('--setups', nargs='+', default=['ball_cross_template', 'ball_within_template'])
p.add_argument('--fold', type=int, default=0)
p.add_argument('--train-tags', nargs='+', default=['_sub400'])
p.add_argument('--save-preds', default='')
p.add_argument('--calib', choices=['task', 'template_cv'], default='task')
p.add_argument('--cv-groups', type=int, default=4)
p.add_argument('--save-probe', default='')
p.add_argument('--dev-tag', default='_n100')
p.add_argument('--grids', nargs='+', default=['full', 'pooled7'])
p.add_argument('--epochs', type=int, default=8)
p.add_argument('--calib-frac', type=float, default=0.2)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--max-rollouts', type=int, default=0)
p.add_argument('--out', required=True)
args = p.parse_args()
dev = torch.device('cuda')
W = 256
GREEN, BLUE, PURPLE, BLACK = 9, 10, 11, 13
SHAPES = ['ball', 'bar', 'jar', 'sticks']
meta = json.load(open(sorted(glob.glob(os.path.join(args.latents, 'meta_shard*.json')))[0]))
GF, G = meta['grid_frames'], meta['grid']
T = len(GF)


def load(manifest):
    names = manifest if isinstance(manifest, list) else [manifest]
    rows = [json.loads(l) for m in names for l in open(os.path.join(args.data, 'manifests', m + '.jsonl'))]
    if args.max_rollouts:
        rows = rows[:args.max_rollouts]
    lat = torch.from_numpy(np.stack([np.load(os.path.join(args.latents, os.path.basename(r['file']).replace('.npz', '.npy'))) for r in rows]))
    feats = [np.load(os.path.join(args.data, r['file']))['features'][GF] for r in rows]
    return rows, lat, feats


def dynamic(f):
    return [o for o in range(f.shape[1]) if f[0, o, BLACK] < .5 and f[0, o, PURPLE] < .5]


def samples(idx, feats):
    out = []
    for r in idx:
        f = feats[r]
        for o in dynamic(f):
            for t in range(T):
                out.append((r, t, o))
    return np.array(out)


class Probe(nn.Module):
    def __init__(self, D, S, d=256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.space = nn.Parameter(torch.zeros(1, S, d))
        self.qin = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.Linear(d, d))
        self.att = nn.ModuleList([nn.MultiheadAttention(d, 4, batch_first=True) for _ in range(2)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(2)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.out = nn.Linear(d, 4)

    def forward(self, tok, q):
        kv = self.inp(tok.float()) + self.space
        h = self.qin(q)[:, None]
        for att, mlp, ln in zip(self.att, self.mlp, self.ln):
            h = h + att(ln(h), kv, kv)[0]
            h = h + mlp(h)
        o = self.out(h[:, 0])
        return torch.sigmoid(o[:, :2]) * W, torch.tanh(o[:, 2:])


def tokens(lat, S_idx, grid):
    x = lat[S_idx[:, 0], S_idx[:, 1]].to(dev, non_blocking=True)
    if grid == 'pooled7':
        N, S, D = x.shape
        x = F.adaptive_avg_pool2d(x.reshape(N, G, G, D).permute(0, 3, 1, 2).float(), 7).permute(0, 2, 3, 1).reshape(N, 49, D)
    return x


def targets(S_idx, feats):
    q = np.stack([feats[r][0, o] for r, t, o in S_idx])
    xy = np.stack([feats[r][t, o, :2] * W for r, t, o in S_idx])
    a = np.stack([feats[r][t, o, 2] * 2 * np.pi for r, t, o in S_idx])
    ang = np.stack([np.cos(2 * a), np.sin(2 * a)], -1)
    bar = np.array([np.argmax(feats[r][0, o, 4:8]) == 1 for r, t, o in S_idx])
    return [torch.from_numpy(v.astype(np.float32)) for v in (q, xy, ang, bar)]


def train_probe(lat, feats, idx, grid):
    torch.manual_seed(args.seed)
    S_all = samples(idx, feats)
    q, xy, ang, bar = targets(S_all, feats)
    S_ = 49 if grid == 'pooled7' else G * G
    model = Probe(lat.shape[-1], S_).to(dev)
    steps = args.epochs * (len(S_all) // 512 + 1)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    for _ in range(args.epochs):
        for b in torch.randperm(len(S_all)).split(512):
            pxy, pang = model(tokens(lat, S_all[b.numpy()], grid), q[b].to(dev))
            loss = F.l1_loss(pxy, xy[b].to(dev)) / W * 10 + (F.l1_loss(pang, ang[b].to(dev), reduction='none').mean(-1) * bar[b].to(dev)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def predict_feats(model, lat, feats, idx, grid):
    out = {}
    for r in idx:
        f = feats[r].copy()
        dyn = dynamic(f)
        if dyn:
            S_idx = np.array([(r, t, o) for o in dyn for t in range(T)])
            q = torch.from_numpy(np.stack([feats[r][0, o] for _, _, o in S_idx]).astype(np.float32)).to(dev)
            pxy, pang = model(tokens(lat, S_idx, grid), q)
            pxy, pang = pxy.cpu().numpy(), pang.cpu().numpy()
            for k, (_, t, o) in enumerate(S_idx):
                f[t, o, :2] = pxy[k] / W
                f[t, o, 2] = (np.arctan2(pang[k, 1], pang[k, 0]) / 2) / (2 * np.pi) % 1
        for o in range(f.shape[1]):
            if o not in dyn:
                f[:, o] = f[0, o]
        out[r] = f
    return out


def surface_dist(f):
    g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
    goals = np.flatnonzero((f[0, :, BLUE] > .5) | (f[0, :, PURPLE] > .5))
    cg, rg = f[:, g, :2] * W, f[0, g, 3] * W / 2
    ds = []
    for o in goals:
        co, L = f[:, o, :2] * W, f[0, o, 3] * W
        if np.argmax(f[0, o, 4:8]) == 1:
            th = f[:, o, 2] * 2 * np.pi
            u = np.stack([np.cos(th), np.sin(th)], -1)
            t = np.clip(((cg - co) * u).sum(-1), -L / 2, L / 2)
            ds.append(np.linalg.norm(cg - (co + t[:, None] * u), axis=-1) - rg)
        else:
            ds.append(np.linalg.norm(cg - co, axis=-1) - rg - L / 2)
    return np.min(ds, 0)


def longest(m):
    b = r = 0
    for v in m:
        r = r + 1 if v else 0
        b = max(b, r)
    return b


RULES = [('hard', tau, 0) for tau in [1, 2, 3, 4, 6, 8, 12, 16]] + [('soft', tau, s) for tau in [2, 4, 6, 8, 12, 16] for s in [1, 2, 4, 8]]


def rule_scores(D, rule):
    kind, tau, s = rule
    if kind == 'hard':
        return np.array([longest(d < tau) for d in D])
    return np.array([(0.5 * (1 + np.tanh((tau - d) / (2 * s)))).sum() for d in D])


def obj_array(xs):
    a = np.empty(len(xs), dtype=object)
    for i, x in enumerate(xs):
        a[i] = x
    return a


def within_task_auc(tasks, y, sc):
    wins = n = 0
    for t in set(tasks):
        m = np.array([x == t for x in tasks])
        pos, neg = sc[m & (y > 0)], sc[m & (y == 0)]
        for a in pos:
            wins += (a > neg).sum() + 0.5 * (a == neg).sum()
            n += len(neg)
    return round(float(wins / n), 4) if n else None


def select_and_eval(Dcal, ycal, Ddev, ydev, tdev):
    best = max(RULES, key=lambda r: roc_auc_score(ycal, rule_scores(Dcal, r)))
    best_hard = max([r for r in RULES if r[0] == 'hard'], key=lambda r: roc_auc_score(ycal, rule_scores(Dcal, r)))
    sc = rule_scores(Ddev, best)
    return {'rule': list(best), 'auc': round(float(roc_auc_score(ydev, sc)), 4), 'within_task_auc': within_task_auc(tdev, ydev, sc),
            'hard_rule': list(best_hard), 'hard_auc': round(float(roc_auc_score(ydev, rule_scores(Ddev, best_hard))), 4),
            'dev_scores': [round(float(v), 4) for v in sc]}


t0 = time.time()
res = {'latents': args.latents, 'encoder': meta['encoder'], 'chunk': meta['chunk'], 'grid': G, 'time_points': T}
for setup in args.setups:
    rtr, Ltr, Ftr = load([f'{setup}_f{args.fold}_train{t}' for t in args.train_tags])
    rdv, Ldv, Fdv = load(f'{setup}_f{args.fold}_dev{args.dev_tag}')
    tasks = sorted({r['task'] for r in rtr})
    rng = np.random.default_rng(args.seed)
    if args.calib == 'task':
        cal_tasks = set(rng.choice(tasks, int(len(tasks) * args.calib_frac), replace=False))
        i_cal = [i for i, r in enumerate(rtr) if r['task'] in cal_tasks]
        i_fit = [i for i, r in enumerate(rtr) if r['task'] not in cal_tasks]
        groups = None
    else:
        tpls = sorted({r['task'][:5] for r in rtr})
        has_blue = {t for i, r in enumerate(rtr) if (Ftr[i][0, :, BLUE] > .5).any() for t in [r['task'][:5]]}
        order = [t for t in tpls if t in has_blue] + [t for t in tpls if t not in has_blue]
        groups = [order[g::args.cv_groups] for g in range(args.cv_groups)]
        i_cal = list(range(len(rtr)))
        i_fit = i_cal
    ycal = np.array([rtr[i]['solved'] for i in i_cal]); ydv = np.array([r['solved'] for r in rdv]); tdv = [r['task'] for r in rdv]
    ref = select_and_eval([surface_dist(Ftr[i]) for i in i_cal], ycal, [surface_dist(f) for f in Fdv], ydv, tdv)
    r = {'n_fit': len(i_fit), 'n_cal': len(i_cal), 'n_dev': len(rdv), 'reference_true_states': ref}
    g_base = []
    for f in Fdv:
        g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
        g_base.append(np.linalg.norm((f[:, g, :2] - f[0, g, :2]) * W, axis=-1))
    r['green_initial_position_baseline_median_px'] = round(float(np.median(np.concatenate(g_base))), 2)
    if groups:
        r['cv_groups'] = groups
    for grid in args.grids:
        if groups:
            Pcal = {}
            for gset in groups:
                held = [i for i, x in enumerate(rtr) if x['task'][:5] in gset]
                fit = [i for i, x in enumerate(rtr) if x['task'][:5] not in gset]
                Pcal.update(predict_feats(train_probe(Ltr, Ftr, fit, grid), Ltr, Ftr, held, grid))
            model = train_probe(Ltr, Ftr, i_fit, grid)
        else:
            model = train_probe(Ltr, Ftr, i_fit, grid)
            Pcal = predict_feats(model, Ltr, Ftr, i_cal, grid)
        Pdv = predict_feats(model, Ldv, Fdv, range(len(rdv)), grid)
        errs, dyn_errs = [], []
        for k, f in enumerate(Fdv):
            g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
            errs.append(np.linalg.norm((Pdv[k][:, g, :2] - f[:, g, :2]) * W, axis=-1))
            for o in dynamic(f):
                dyn_errs.append(np.linalg.norm((Pdv[k][:, o, :2] - f[:, o, :2]) * W, axis=-1))
        e = np.concatenate(errs)
        e2e = select_and_eval([surface_dist(Pcal[i]) for i in i_cal], ycal, [surface_dist(Pdv[k]) for k in range(len(rdv))], ydv, tdv)
        r[grid] = {'green_err_median_px': round(float(np.median(e)), 2), 'green_err_p90_px': round(float(np.percentile(e, 90)), 2),
                   'all_dynamic_err_median_px': round(float(np.median(np.concatenate(dyn_errs))), 2),
                   'end_to_end': e2e, 'pass': e2e['auc'] >= ref['auc'] - 0.03}
        if args.save_probe:
            os.makedirs(args.save_probe, exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'rule': e2e['rule'], 'grid': grid, 'setup': setup, 'fold': args.fold,
                        'calib': args.calib, 'latents': args.latents, 'train_tags': args.train_tags, 'D': Ltr.shape[-1], 'S': 49 if grid == 'pooled7' else G * G},
                       os.path.join(args.save_probe, f'{setup}_f{args.fold}_{grid}.pt'))
        if args.save_preds:
            os.makedirs(args.save_preds, exist_ok=True)
            np.savez_compressed(os.path.join(args.save_preds, f'{setup}_{grid}.npz'), pred=obj_array([Pdv[k] for k in range(len(rdv))]),
                                files=np.array([r['file'] for r in rdv]), solved=ydv, tasks=np.array(tdv),
                                e2e_scores=np.array(e2e['dev_scores']), ref_scores=np.array(ref['dev_scores']))
        print(setup, grid, json.dumps({k: v for k, v in r[grid].items() if k != 'end_to_end'} | {'end_to_end': {k: v for k, v in e2e.items() if k != 'dev_scores'}}), flush=True)
    r['reference_auc'] = ref['auc']
    res[setup] = r
    print(setup, 'reference', json.dumps(ref), 'baseline', r['green_initial_position_baseline_median_px'], flush=True)
res['n_train_rollouts'] = {s: res[s]['n_fit'] + res[s]['n_cal'] for s in args.setups}
res[f'PASS_cross_f{args.fold}'] = {g: res.get('ball_cross_template', {}).get(g, {}).get('pass') for g in args.grids}
res['seconds'] = round(time.time() - t0, 1)
res['args'] = vars(args)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, 'w'), indent=1)
print(json.dumps({k: res[k] for k in [f'PASS_cross_f{args.fold}', 'seconds']}))
