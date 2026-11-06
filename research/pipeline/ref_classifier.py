import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument('--data', default='/path/to/phyre-unc/data')
p.add_argument('--setups', nargs='+', default=['ball_within_template', 'ball_cross_template'])
p.add_argument('--fold', type=int, default=0)
p.add_argument('--dev-tag', default='_n100')
p.add_argument('--train-tag', default='')
p.add_argument('--arch', choices=['pool', 'relational'], default='relational')
p.add_argument('--epochs', type=int, default=40)
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--out', default='/path/to/phyre-unc/results/step2_reference.json')
args = p.parse_args()
dev = torch.device('cuda')
GREEN = 9
W = 256


def load(manifest):
    rows = [json.loads(l) for l in open(os.path.join(args.data, 'manifests', manifest + '.jsonl'))]
    feats = [np.load(os.path.join(args.data, r['file']))['features'] for r in rows]
    n = max(f.shape[1] for f in feats)
    X = np.zeros((len(feats), feats[0].shape[0], n, 14), np.float32)
    M = np.zeros((len(feats), n), bool)
    for i, f in enumerate(feats):
        X[i, :, :f.shape[1]] = f
        M[i, :f.shape[1]] = True
    y = np.array([r['solved'] for r in rows], np.float32)
    return X, M, y, rows


class Head(nn.Module):
    def __init__(self, T, d=128):
        super().__init__()
        self.obj = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.Linear(d, d))
        self.time = nn.Parameter(torch.zeros(1, T, d))
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), 2)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Linear(d, 1)

    def forward(self, x, m):
        h = self.obj(x)
        mf = m[:, None, :, None].float()
        tok = (h * mf).sum(2) / mf.sum(2) + (h - 1e4 * (1 - mf)).amax(2)
        tok = self.temporal(tok + self.time[:, :tok.shape[1]])
        z, _ = self.pool(self.q.expand(len(tok), -1, -1), tok, tok)
        return self.out(z[:, 0]).squeeze(-1)


class RelHead(nn.Module):
    def __init__(self, T, d=128):
        super().__init__()
        self.obj = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.Linear(d, d))
        self.objattn = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), 2)
        self.oq = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.opool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.time = nn.Parameter(torch.zeros(1, T, d))
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), 2)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Linear(d, 1)

    def forward(self, x, m):
        B, T, N, _ = x.shape
        pad = (~m)[:, None, :].expand(B, T, N).reshape(B * T, N)
        h = self.objattn(self.obj(x).reshape(B * T, N, -1), src_key_padding_mask=pad)
        tok, _ = self.opool(self.oq.expand(B * T, -1, -1), h, h, key_padding_mask=pad)
        tok = self.temporal(tok.reshape(B, T, -1) + self.time[:, :T])
        z, _ = self.pool(self.q.expand(B, -1, -1), tok, tok)
        return self.out(z[:, 0]).squeeze(-1)


def train_eval(Xtr, Mtr, ytr, Xdv, Mdv, ydv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = (RelHead if args.arch == 'relational' else Head)(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xtr, Mtr, ytr = map(lambda a: torch.from_numpy(a).to(dev), (Xtr, Mtr, ytr))
    Xdv, Mdv = map(lambda a: torch.from_numpy(a).to(dev), (Xdv, Mdv))
    for ep in range(args.epochs):
        model.train()
        for idx in torch.randperm(len(ytr), device=dev).split(64):
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xtr[idx], Mtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        s = torch.cat([model(Xdv[i:i + 256], Mdv[i:i + 256]) for i in range(0, len(Xdv), 256)]).cpu().numpy()
    return float(roc_auc_score(ydv, s)), float(((s > 0) == (ydv > 0.5)).mean())


def green_initial_baseline(X, M):
    errs = []
    for x, m in zip(X, M):
        g = np.flatnonzero(m & (x[0, :, GREEN] > 0.5))
        if len(g):
            xy = x[:, g[0], :2] * W
            errs.append(np.linalg.norm(xy - xy[0], axis=-1))
    e = np.concatenate(errs)
    return {'median_px_all_frames': round(float(np.median(e)), 2), 'mean_px': round(float(e.mean()), 2),
            'p90_px': round(float(np.percentile(e, 90)), 2), 'frac_frames_over_9px': round(float((e > 9).mean()), 3)}


t0 = time.time()
res = {}
for setup in args.setups:
    Xtr, Mtr, ytr, _ = load(f'{setup}_f{args.fold}_train{args.train_tag}')
    Xdv, Mdv, ydv, _ = load(f'{setup}_f{args.fold}_dev{args.dev_tag}')
    r = {'n_train': len(ytr), 'n_dev': len(ydv), 'dev_pos_rate': float(ydv.mean()),
         'green_initial_position_baseline_dev': green_initial_baseline(Xdv, Mdv)}
    for name, sl in [('full_trajectory', slice(None)), ('first_frame_only', slice(0, 1))]:
        runs = [train_eval(Xtr[:, sl], Mtr, ytr, Xdv[:, sl], Mdv, ydv, s) for s in args.seeds]
        r[name] = {'auc': [round(a, 4) for a, _ in runs], 'acc': [round(c, 4) for _, c in runs],
                   'auc_mean': round(float(np.mean([a for a, _ in runs])), 4)}
        print(setup, name, r[name], flush=True)
    res[setup] = r
res['gate_within_dev_auc>=0.98'] = res.get('ball_within_template', {}).get('full_trajectory', {}).get('auc_mean', 0) >= 0.98
res['seconds'] = round(time.time() - t0, 1)
res['args'] = vars(args)
res['arch'] = args.arch
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, 'w'), indent=1)
print(json.dumps(res, indent=1))
