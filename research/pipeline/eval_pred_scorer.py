import argparse, json, os
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument('--probe', required=True)
p.add_argument('--real', default='/mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32')
p.add_argument('--preds', required=True)
p.add_argument('--data', default='/path/to/phyre-unc/data')
p.add_argument('--dev', default='ball_within_template_f0_dev_n100')
p.add_argument('--out', required=True)
args = p.parse_args()
dev = torch.device('cuda')
W, GF = 256, list(range(0, 128, 4))
GREEN, BLUE, PURPLE, BLACK = 9, 10, 11, 13


class AttnProbe(nn.Module):
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


def dynamic(f):
    return [o for o in range(f.shape[1]) if f[0, o, BLACK] < .5 and f[0, o, PURPLE] < .5]


@torch.no_grad()
def probe_feats(model, lat, f0):
    f = f0.copy()
    dyn = dynamic(f0)
    for o in range(f.shape[1]):
        if o not in dyn:
            f[:, o] = f0[0, o]
    if dyn:
        tok = torch.from_numpy(lat).to(dev)
        for o in dyn:
            q = torch.from_numpy(np.repeat(f0[:1, o], len(GF), 0).astype(np.float32)).to(dev)
            pxy, pang = model(tok, q)
            f[:, o, :2] = pxy.cpu().numpy() / W
            f[:, o, 2] = (np.arctan2(pang[:, 1].cpu().numpy(), pang[:, 0].cpu().numpy()) / 2) / (2 * np.pi) % 1
    return f


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


def score(d, rule):
    kind, tau, s = rule
    return longest(d < tau) if kind == 'hard' else float((0.5 * (1 + np.tanh((tau - d) / (2 * s)))).sum())


def within_task_auc(tasks, y, sc):
    wins = n = 0
    for t in set(tasks):
        m = tasks == t
        for a in sc[m & (y > 0)]:
            neg = sc[m & (y == 0)]
            wins += (a > neg).sum() + 0.5 * (a == neg).sum()
            n += len(neg)
    return round(float(wins / n), 4)


ck = torch.load(args.probe, map_location='cpu', weights_only=False)
model = AttnProbe(ck['D'], ck['S']).to(dev)
model.load_state_dict(ck['state_dict'])
model.eval()
rows = [json.loads(l) for l in open(os.path.join(args.data, 'manifests', args.dev + '.jsonl'))]
y = np.array([r['solved'] for r in rows]); tasks = np.array([r['task'] for r in rows])
res = {'probe': args.probe, 'rule': ck['rule'], 'preds': args.preds}
green_err = {'real': [], 'pred': []}
scores = {'true_states': [], 'real': [], 'pred': []}
for r in rows:
    name = os.path.basename(r['file']).replace('.npz', '.npy')
    f = np.load(os.path.join(args.data, r['file']))['features'][GF]
    g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
    scores['true_states'].append(score(surface_dist(f), ck['rule']))
    real = np.load(os.path.join(args.real, name))
    pred = np.load(os.path.join(args.preds, name))
    pred[0] = real[0]
    for key, lat in [('real', real), ('pred', pred)]:
        pf = probe_feats(model, lat, f)
        scores[key].append(score(surface_dist(pf), ck['rule']))
        green_err[key].append(np.linalg.norm((pf[:, g, :2] - f[:, g, :2]) * W, axis=-1))
for key, sc in scores.items():
    sc = np.array(sc)
    res[f'auc_{key}'] = round(float(roc_auc_score(y, sc)), 4)
    res[f'within_task_auc_{key}'] = within_task_auc(tasks, y, sc)
for key, e in green_err.items():
    e = np.stack(e)
    res[f'green_err_{key}_median_px'] = round(float(np.median(e)), 2)
    res[f'green_err_{key}_by_time_median_px'] = [round(float(v), 1) for v in np.median(e, 0)]
json.dump(res, open(args.out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if 'by_time' not in k}, indent=1))
print('green err by time (real):', res['green_err_real_by_time_median_px'])
print('green err by time (pred):', res['green_err_pred_by_time_median_px'])
