import argparse, json, os, time
from collections import defaultdict
import numpy as np
import torch
from common import Predictor, MemoryPredictor

p = argparse.ArgumentParser()
p.add_argument('--packs', default='/mnt/meg/jjenchen/phyre-unc/packs')
p.add_argument('--train', default='within_f0_train_s16r16_pca128')
p.add_argument('--base', default='/mnt/meg/jjenchen/phyre-unc/runs/base_s16r16_pca128/predictor.pt')
p.add_argument('--mode', choices=['full', 'outcome', 'action', 'residual'], default='full')
p.add_argument('--steps', type=int, default=20000)
p.add_argument('--batch', type=int, default=8)
p.add_argument('--kmax', type=int, default=4)
p.add_argument('--p-empty', type=float, default=0.1)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--out', required=True)
p.add_argument('--mem-steps', type=int, default=8)
p.add_argument('--mem-grid', type=int, default=7)
p.add_argument('--per-attempt', action='store_true')
p.add_argument('--unfreeze', action='store_true')
p.add_argument('--backbone-lr', type=float, default=3e-5)
p.add_argument('--gate-empty', action='store_true')
args = p.parse_args()
dev = torch.device('cuda')
torch.manual_seed(0)
rng = np.random.default_rng(0)
t0 = time.time()
lat = np.load(os.path.join(args.packs, args.train + '_latents.npy'), mmap_mode='r')
meta = np.load(os.path.join(args.packs, args.train + '_meta.npz'))
A, Y, TK = meta['action'], meta['solved'].astype(np.int64), meta['tasks']
by = defaultdict(list)
for i, t in enumerate(TK):
    by[t].append(i)
tasks = [t for t in by if len(by[t]) >= 2]
pc = torch.load(args.base, map_location='cpu', weights_only=False); pa = pc['args']
base = Predictor(pc['Din'], pc['T'], pc['S'], pa['d'], pa['layers'], pa['heads'])
base.load_state_dict(pc['state_dict'])
model = MemoryPredictor(base, pc['Din'], mode=args.mode, mem_steps=args.mem_steps, mem_grid=args.mem_grid, per_attempt=args.per_attempt, freeze_base=not args.unfreeze, gate_empty=args.gate_empty).to(dev)
params = [q for q in model.parameters() if q.requires_grad]
mem_params = [q for n, q in model.named_parameters() if q.requires_grad and not n.startswith('base.')]
base_params = [q for n, q in model.named_parameters() if q.requires_grad and n.startswith('base.')]
groups = [{'params': mem_params, 'lr': args.lr}] + ([{'params': base_params, 'lr': args.backbone_lr}] if base_params else [])
opt = torch.optim.AdamW(groups, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g['lr'] for g in groups], total_steps=args.steps, pct_start=0.03)
print(f'loaded in {time.time() - t0:.0f}s; trainable {sum(q.numel() for q in params) / 1e6:.2f}M; tasks {len(tasks)}', flush=True)
T_ = lat.shape[1]
log, run = [], 0.0
for step in range(args.steps):
    qi, ci, cm = [], np.zeros((args.batch, args.kmax), np.int64), np.zeros((args.batch, args.kmax), bool)
    for b in range(args.batch):
        idx = by[tasks[rng.integers(len(tasks))]]
        k = 0 if rng.random() < args.p_empty else int(rng.integers(1, min(args.kmax, len(idx) - 1) + 1))
        pick = rng.choice(idx, k + 1, replace=False)
        qi.append(pick[0]); ci[b, :k] = pick[1:]; cm[b, :k] = True
    qi = np.array(qi)
    zq = torch.from_numpy(np.stack([lat[i] for i in qi])).to(dev).float()
    need = sorted(set(ci[cm].tolist()))
    full = args.mode == 'residual'
    cache = {i: (lat[i] if full else lat[i, ::T_ // args.mem_steps]) for i in need}
    ctx = np.zeros((args.batch, args.kmax) + ((T_,) if full else (args.mem_steps,)) + lat.shape[2:], np.float16)
    for b in range(args.batch):
        for j in range(args.kmax):
            if cm[b, j]:
                ctx[b, j] = cache[ci[b, j]]
    ctx = torch.from_numpy(ctx).to(dev).float()
    ca = torch.from_numpy(A[ci]).to(dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        res = model.context_residual(ctx, ca).float() if full else None
        mem, mm = model.encode_memory(ctx, ca, torch.from_numpy(Y[ci]).to(dev), torch.from_numpy(cm).to(dev), res)
        pr = model(zq[:, 0], torch.from_numpy(A[qi]).to(dev), mem, mm)
    loss = (pr.float()[:, 1:] - zq[:, 1:]).abs().mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step(); run += loss.item()
    if (step + 1) % 1000 == 0:
        rec = {'step': step + 1, 'train_l1': round(run / 1000, 5), 'seconds': round(time.time() - t0)}
        log.append(rec); run = 0.0
        print(json.dumps(rec), flush=True)
os.makedirs(args.out, exist_ok=True)
torch.save({'state_dict': {k: v for k, v in model.state_dict().items() if args.unfreeze or not k.startswith('base.')}, 'mode': args.mode,
            'mem_steps': args.mem_steps, 'mem_grid': args.mem_grid, 'per_attempt': args.per_attempt, 'unfreeze': args.unfreeze, 'gate_empty': args.gate_empty,
            'base': args.base, 'args': vars(args), 'log': log}, os.path.join(args.out, 'memory.pt'))
print(json.dumps({'done': True, 'seconds': round(time.time() - t0)}))
