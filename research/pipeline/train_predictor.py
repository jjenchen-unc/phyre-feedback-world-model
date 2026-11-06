import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--packs', default='/mnt/meg/jjenchen/phyre-unc/packs')
p.add_argument('--train', default='within_f0_train_div16')
p.add_argument('--dev', default='within_f0_dev')
p.add_argument('--motion-beta', type=float, default=0.0)
p.add_argument('--aux-probe', default='')
p.add_argument('--aux-lambda', type=float, default=0.0)
p.add_argument('--d', type=int, default=384)
p.add_argument('--layers', type=int, default=6)
p.add_argument('--heads', type=int, default=6)
p.add_argument('--epochs', type=int, default=20)
p.add_argument('--batch', type=int, default=8)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--max-rollouts', type=int, default=0)
p.add_argument('--out', required=True)
args = p.parse_args()
dev = torch.device('cuda')
torch.manual_seed(0)


def load(name, in_ram):
    lat = np.load(os.path.join(args.packs, name + '_latents.npy'), mmap_mode='r')
    meta = dict(np.load(os.path.join(args.packs, name + '_meta.npz')))
    n = args.max_rollouts or len(lat)
    lat = torch.from_numpy(np.ascontiguousarray(lat[:n])) if in_ram else lat
    return meta, lat, torch.from_numpy(meta['action'][:n]), torch.from_numpy(meta['features'][:n]), torch.from_numpy(meta['mask'][:n])


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
        return torch.sigmoid(self.out(h[:, 0])[:, :2]) * 256


def aux_loss(probe, pred, feats, mask):
    B, T = pred.shape[:2]
    dyn = mask & (feats[:, 0, :, 13] < .5) & (feats[:, 0, :, 11] < .5)
    b, o = torch.nonzero(dyn, as_tuple=True)
    tok = pred[b, 1:].reshape(len(b) * (T - 1), pred.shape[2], pred.shape[3])
    q = feats[b, 0, o][:, None].expand(-1, T - 1, -1).reshape(-1, 14)
    xy = (feats[b, 1:, o, :2] * 256).reshape(-1, 2)
    return F.l1_loss(probe(tok, q), xy) / 256 * 10


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.qkv, self.o = h, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sp, self.tm = Attn(d, h), Attn(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, S, D = x.shape
        x = x + self.sp(self.n1(x).reshape(B * T, S, D)).reshape(B, T, S, D)
        y = self.n2(x).permute(0, 2, 1, 3).reshape(B * S, T, D)
        x = x + self.tm(y).reshape(B, S, T, D).permute(0, 2, 1, 3)
        return x + self.mlp(self.n3(x))


class Predictor(nn.Module):
    def __init__(self, Din, T, S, d, L, h):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(Din), nn.Linear(Din, d))
        self.act = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.time = nn.Parameter(torch.zeros(1, T, 1, d))
        self.space = nn.Parameter(torch.zeros(1, 1, S, d))
        nn.init.trunc_normal_(self.time, std=0.02); nn.init.trunc_normal_(self.space, std=0.02)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, Din))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)
        self.T = T

    def forward(self, z0, a):
        x = self.inp(z0)[:, None] + self.time + self.space + self.act(a)[:, None, None]
        for b in self.blocks:
            x = b(x)
        out = z0[:, None] + self.out(x)
        return torch.cat([z0[:, None], out[:, 1:]], 1)


@torch.no_grad()
def evaluate(model, lat, act):
    model.eval()
    err_m, err_c = torch.zeros(lat.shape[1]), torch.zeros(lat.shape[1])
    preds = []
    for i in range(0, len(lat), args.batch):
        z = lat[i:i + args.batch].to(dev).float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pr = model(z[:, 0], act[i:i + args.batch].to(dev)).float()
        err_m += (pr - z).abs().mean((2, 3)).sum(0).cpu()
        err_c += (z[:, :1] - z).abs().mean((2, 3)).sum(0).cpu()
        preds.append(pr.half().cpu())
    model.train()
    return err_m / len(lat), err_c / len(lat), torch.cat(preds)


t0 = time.time()
os.makedirs(args.out, exist_ok=True)
_, Ltr, Atr, Ftr, Mtr = load(args.train, in_ram=False)
mdv, Ldv, Adv, _, _ = load(args.dev, in_ram=True)
probe = None
if args.aux_probe:
    ck = torch.load(args.aux_probe, map_location='cpu', weights_only=False)
    probe = AttnProbe(ck['D'], ck['S']).to(dev)
    probe.load_state_dict({k: v for k, v in ck['state_dict'].items()}, strict=True)
    probe.eval().requires_grad_(False)
print(f'loaded train {tuple(Ltr.shape)} dev {tuple(Ldv.shape)} in {time.time() - t0:.0f}s', flush=True)
T, S, Din = Ltr.shape[1:]
N = len(Atr)
model = Predictor(Din, T, S, args.d, args.layers, args.heads).to(dev)
opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.05)
steps = args.epochs * (N // args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.03)
log = []
for ep in range(args.epochs):
    tot = 0.0
    for b in torch.randperm(N).split(args.batch)[:N // args.batch]:
        b = b.sort().values
        z = torch.from_numpy(np.ascontiguousarray(Ltr[b.numpy()])).to(dev, non_blocking=True).float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pr = model(z[:, 0], Atr[b].to(dev))
        err = (pr.float()[:, 1:] - z[:, 1:]).abs()
        if args.motion_beta > 0:
            m = (z[:, 1:] - z[:, :1]).abs().mean(-1, keepdim=True)
            loss = (err * (1 + args.motion_beta * m / m.mean())).mean()
        else:
            loss = err.mean()
        if probe is not None:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = loss + args.aux_lambda * aux_loss(probe, pr, Ftr[b].to(dev), Mtr[b].to(dev))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tot += loss.item()
    em, ec, _ = evaluate(model, Ldv, Adv)
    rec = {'epoch': ep + 1, 'train_l1': round(tot / (N // args.batch), 4), 'dev_l1_model': round(float(em[1:].mean()), 4),
           'dev_l1_copy_baseline': round(float(ec[1:].mean()), 4), 'seconds': round(time.time() - t0)}
    log.append(rec)
    print(json.dumps(rec), flush=True)
em, ec, preds = evaluate(model, Ldv, Adv)
torch.save({'state_dict': model.state_dict(), 'args': vars(args), 'T': T, 'S': S, 'Din': Din}, os.path.join(args.out, 'predictor.pt'))
pdir = os.path.join(args.out, 'dev_preds')
os.makedirs(pdir, exist_ok=True)
for f, pr in zip(mdv['files'], preds):
    np.save(os.path.join(pdir, os.path.basename(str(f)).replace('.npz', '.npy')), pr.numpy())
summary = {'per_time_dev_l1_model': [round(float(v), 4) for v in em], 'per_time_dev_l1_copy': [round(float(v), 4) for v in ec],
           'dev_l1_model_mean': round(float(em[1:].mean()), 4), 'dev_l1_copy_mean': round(float(ec[1:].mean()), 4),
           'n_train': N, 'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 2), 'log': log,
           'seconds': round(time.time() - t0), 'args': vars(args)}
json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
print(json.dumps({k: summary[k] for k in ['dev_l1_model_mean', 'dev_l1_copy_mean', 'params_M', 'seconds']}))
