import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument('--latents', required=True)
p.add_argument('--data', default='/path/to/phyre-unc/data')
p.add_argument('--setups', nargs='+', default=['ball_cross_template', 'ball_within_template'])
p.add_argument('--fold', type=int, default=0)
p.add_argument('--train-tag', default='_sub400')
p.add_argument('--dev-tag', default='_n100')
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--reference', default='/path/to/phyre-unc/results/step3_reference_sub400.json')
p.add_argument('--out', required=True)
args = p.parse_args()
dev = torch.device('cuda')
GREEN, W = 9, 256
meta = json.load(open(sorted(f for f in [os.path.join(args.latents, x) for x in os.listdir(args.latents)] if os.path.basename(f).startswith('meta_'))[0]))
PF, G = meta['probe_frames'], meta['grid']


def load(manifest):
    rows = [json.loads(l) for l in open(os.path.join(args.data, 'manifests', manifest + '.jsonl'))]
    pooled, probe, xy, y = [], [], [], []
    for r in rows:
        rec = np.load(os.path.join(args.latents, os.path.basename(r['file']).replace('.npz', '.npy')))
        pooled.append(rec['pooled']); probe.append(rec['probe'])
        f = np.load(os.path.join(args.data, r['file']))['features']
        g = int(np.flatnonzero(f[0, :, GREEN] > 0.5)[0])
        xy.append(f[PF, g, :2] * W)
        y.append(r['solved'])
    return (torch.from_numpy(np.stack(pooled)), torch.from_numpy(np.stack(probe)),
            torch.from_numpy(np.stack(xy).astype(np.float32)), torch.tensor(y, dtype=torch.float32))


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mha = nn.MultiheadAttention(d, 4, batch_first=True)

    def forward(self, x):
        return self.mha(self.q.expand(len(x), -1, -1), x, x)[0][:, 0]


class SuccessHead(nn.Module):
    def __init__(self, D, T, S, d=256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.space = nn.Parameter(torch.zeros(1, S, d))
        self.spool = AttnPool(d)
        self.time = nn.Parameter(torch.zeros(1, T, d))
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), 2)
        self.tpool = AttnPool(d)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        B, T, S, D = x.shape
        h = self.inp(x.float()) + self.space
        h = self.spool(h.reshape(B * T, S, -1)).reshape(B, T, -1)
        return self.out(self.tpool(self.temporal(h + self.time))).squeeze(-1)


class PosProbe(nn.Module):
    def __init__(self, D, S, d=256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.space = nn.Parameter(torch.zeros(1, S, d))
        self.pool = AttnPool(d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, x):
        return torch.sigmoid(self.out(self.pool(self.inp(x.float()) + self.space))) * W


def fit(model, X, T, loss_fn, bs, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for _ in range(args.epochs):
        model.train()
        for idx in torch.randperm(len(X)).split(bs):
            loss = loss_fn(model(X[idx].to(dev)), T[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, X, bs=64):
    return torch.cat([model(X[i:i + bs].to(dev)).float().cpu() for i in range(0, len(X), bs)])


def pool_probe(x):
    N, S, D = x.shape
    return F.adaptive_avg_pool2d(x.reshape(N, G, G, D).permute(0, 3, 1, 2).float(), 7).permute(0, 2, 3, 1).reshape(N, 49, D).half()


t0 = time.time()
ref = json.load(open(args.reference))
res = {'latents': args.latents, 'meta': {k: meta[k] for k in ['encoder', 'chunk', 'res', 'tubelet', 'grid']}}
for setup in args.setups:
    Ptr, Qtr, XYtr, ytr = load(f'{setup}_f{args.fold}_train{args.train_tag}')
    Pdv, Qdv, XYdv, ydv = load(f'{setup}_f{args.fold}_dev{args.dev_tag}')
    r = {'n_train': len(ytr), 'n_dev': len(ydv)}
    aucs = []
    for s in args.seeds:
        m = fit(SuccessHead(Ptr.shape[-1], Ptr.shape[1], Ptr.shape[2]).to(dev), Ptr, ytr, F.binary_cross_entropy_with_logits, 32, s)
        aucs.append(float(roc_auc_score(ydv.numpy(), predict(m, Pdv).numpy())))
    r['success_auc'] = [round(a, 4) for a in aucs]
    r['success_auc_mean'] = round(float(np.mean(aucs)), 4)
    base = (XYdv - XYdv[:, :1]).norm(dim=-1).flatten()
    r['position_baseline_median_px'] = round(float(base.median()), 2)
    Xtr_f, Xdv_f = Qtr.flatten(0, 1), Qdv.flatten(0, 1)
    Ttr, Tdv = XYtr.flatten(0, 1), XYdv.flatten(0, 1)
    for name, (A, B) in {'full_grid': (Xtr_f, Xdv_f), 'pooled_7x7': (pool_probe(Xtr_f), pool_probe(Xdv_f))}.items():
        meds = []
        for s in args.seeds:
            m = fit(PosProbe(A.shape[-1], A.shape[1]).to(dev), A, Ttr, F.l1_loss, 128, s)
            meds.append(float((predict(m, B) - Tdv).norm(dim=-1).median()))
        r[f'position_median_px_{name}'] = [round(v, 2) for v in meds]
        r[f'position_median_px_{name}_mean'] = round(float(np.mean(meds)), 2)
    ref_auc = ref[setup]['full_trajectory']['auc_mean']
    r['reference_auc_mean'] = ref_auc
    pos = r['position_median_px_full_grid_mean']
    r['pass_success'] = r['success_auc_mean'] >= 0.95 and r['success_auc_mean'] >= ref_auc - 0.03
    r['pass_position'] = pos <= 9 and pos <= 0.5 * r['position_baseline_median_px']
    r['pass'] = r['pass_success'] and r['pass_position']
    res[setup] = r
    print(setup, json.dumps(r), flush=True)
res['PASS_cross_dev'] = res.get('ball_cross_template', {}).get('pass')
res['seconds'] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, 'w'), indent=1)
print(json.dumps(res, indent=1))
