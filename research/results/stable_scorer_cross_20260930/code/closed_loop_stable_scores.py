"""Matched legacy/stable scoring evaluation; no training or shortlist recollection."""
import argparse, copy, json, os, time, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import spearmanr
from common import Predictor, AttnProbe, MemoryPredictor, surface_dist, rule_score

p = argparse.ArgumentParser()
p.add_argument('--pack', default='/mnt/meg/jjenchen/phyre-unc/packs/within_f0_dev_shortlist512_pca128')
p.add_argument('--predictor', default='/mnt/meg/jjenchen/phyre-unc/runs/base_s16r16_pca128/predictor.pt')
p.add_argument('--probe', default='/path/to/phyre-unc/results/probes_pca128/ball_within_template_f0_full.pt')
p.add_argument('--memory', default='')
p.add_argument('--feedback', action='store_true')
p.add_argument('--max-updates', type=int, default=10)
p.add_argument('--keep', type=int, default=4)
p.add_argument('--max-tasks', type=int, default=0)
p.add_argument('--out', required=True)
p.add_argument('--score-mode', choices=['legacy', 'stable_log'], required=True)
p.add_argument('--tasks', nargs='+')
p.add_argument('--tta-steps', type=int, default=0)
p.add_argument('--tta-lr', type=float, default=1e-4)
p.add_argument('--force-gate-empty', action='store_true')
p.add_argument('--mem-scale', type=float, default=1.0)
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=1)
args = p.parse_args()
assert args.tta_steps == 0, 'No training in this evaluation'
torch.set_num_threads(4)
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False)); tmp.replace(path)
provenance = {k: sha(v) for k, v in [('predictor', args.predictor), ('probe', args.probe), ('meta', args.pack + '_meta.npz')]}
if args.memory: provenance['memory'] = sha(args.memory)
provenance['code'] = sha(__file__)
provenance['common'] = sha(Path(__file__).with_name('common.py'))
parts = Path(args.out + '.tasks'); parts.mkdir(parents=True, exist_ok=True)
dev = torch.device('cuda')
W, G, GREEN, PURPLE, BLACK = 256, 14, 9, 11, 13
WEIGHTS = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))
lat = np.load(args.pack + '_latents.npy', mmap_mode='r')
meta = np.load(args.pack + '_meta.npz')
F_, M_, A_, Y_, TK = meta['features'], meta['mask'], meta['action'], meta['solved'].astype(bool), meta['tasks']
by = defaultdict(list)
for i, t in enumerate(TK):
    by[str(t)].append(i)
pc = torch.load(args.predictor, map_location='cpu', weights_only=False); pa = pc['args']
pred = Predictor(pc['Din'], pc['T'], pc['S'], pa['d'], pa['layers'], pa['heads']).to(dev)
pred.load_state_dict(pc['state_dict']); pred.eval()
qc = torch.load(args.probe, map_location='cpu', weights_only=False)
probe = AttnProbe(qc['D'], qc['S']).to(dev); probe.load_state_dict(qc['state_dict']); probe.eval()
rule = qc['rule']
mem_model = None
if args.memory:
    mc = torch.load(args.memory, map_location='cpu', weights_only=False)
    mem_model = MemoryPredictor(pred, pc['Din'], mode=mc['mode'], mem_steps=mc.get('mem_steps', 8), mem_grid=mc.get('mem_grid', 7), per_attempt=mc.get('per_attempt', False), gate_empty=mc.get('gate_empty', False) or args.force_gate_empty).to(dev)
    assert mc['base'] == args.predictor and not mc.get('unfreeze', False)
    missing, unexpected = mem_model.load_state_dict(mc['state_dict'], strict=False)
    assert not unexpected and all(k.startswith('base.') for k in missing)
    assert mem_model.gate_empty
    mem_model.eval()
    mem_model.out_scale = args.mem_scale
T_full = lat.shape[1]


@torch.no_grad()
def encode_ctx(ctx):
    K = max(len(ctx), 1)
    full = mem_model.mode == 'residual'
    ms = mem_model.mem_steps
    c_lat = torch.zeros((1, K, T_full if full else ms) + lat.shape[2:], device=dev)
    c_act = torch.zeros(1, K, 3, device=dev); c_out = torch.zeros(1, K, device=dev); c_m = torch.zeros(1, K, dtype=torch.bool, device=dev)
    for j, ci in enumerate(ctx):
        c_lat[0, j] = torch.from_numpy(np.ascontiguousarray(lat[ci] if full else lat[ci, ::T_full // ms])).to(dev).float()
        c_act[0, j] = torch.from_numpy(A_[ci]).to(dev); c_out[0, j] = float(Y_[ci]); c_m[0, j] = True
    with torch.autocast('cuda', dtype=torch.bfloat16):
        res = mem_model.context_residual(c_lat, c_act).float() if full else None
        return mem_model.encode_memory(c_lat, c_act, c_out, c_m, res)


@torch.no_grad()
def score_all(idx, ctx, model=None):
    model = model or pred
    mem = encode_ctx(ctx) if mem_model is not None else None
    out = np.zeros(len(idx)); legacy = np.zeros(len(idx)); stable = np.zeros(len(idx))
    for s in range(0, len(idx), 32):
        b = idx[s:s + 32]
        z0 = torch.from_numpy(np.stack([lat[i, 0] for i in b])).to(dev).float()
        a = torch.from_numpy(A_[b]).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pr = (mem_model(z0, a, mem[0].expand(len(b), -1, -1), mem[1].expand(len(b), -1)) if mem is not None else model(z0, a)).float()
        B, T = pr.shape[:2]
        kv = probe.inp(pr.reshape(B * T, G * G, -1)) + probe.space
        for k, i in enumerate(b):
            f = F_[i][:, M_[i]]
            dyn = [o for o in range(f.shape[1]) if f[0, o, BLACK] < .5 and f[0, o, PURPLE] < .5]
            pf = np.repeat(f[:1], T, 0).copy()
            for o in dyn:
                q = torch.from_numpy(np.repeat(f[:1, o], T, 0)).to(dev)
                pxy, pang = probe.read(kv[k * T:(k + 1) * T], q)
                pf[:, o, :2] = pxy.cpu().numpy() / W
                pang = pang.cpu().numpy()
                pf[:, o, 2] = (np.arctan2(pang[:, 1], pang[:, 0]) / 2) / (2 * np.pi) % 1
            distance = surface_dist(pf)
            assert rule[0] == 'soft' and rule[2] > 0
            legacy[s + k] = rule_score(distance, rule)
            margin = (float(rule[1]) - distance.astype(np.float64)) / float(rule[2])
            stable[s + k] = np.logaddexp.reduce(-np.logaddexp(0., -margin))
    assert np.isfinite(legacy).all() and np.isfinite(stable).all()
    out = legacy if args.score_mode == 'legacy' else stable
    score_history.append(np.stack([legacy, stable]))
    context_history.append(list(ctx))
    return out


t0 = time.time()
tasks = sorted(by)[:args.max_tasks] if args.max_tasks else sorted(by)
if args.tasks: tasks = [t for t in tasks if t in args.tasks]; assert len(tasks) == len(args.tasks)
tasks = tasks[args.shard::args.nshards]
per, rhos = [], []
for ti, task in enumerate(tasks):
    part = parts / (task.replace(':', '_') + '.json')
    if part.exists():
        saved = json.loads(part.read_text())
        assert saved['provenance'] == provenance and saved['args'] == vars(args)
        per.append(saved['result']); rhos.append(saved['initial_rho'])
        continue
    score_history, context_history, attempts_trace = [], [], []
    idx = by[task]
    tmodel = copy.deepcopy(pred) if args.tta_steps else None
    fails = []
    sc = score_all(idx, [])
    rho = float(spearmanr(sc, -np.arange(len(idx))).correlation)
    rhos.append(rho if np.isfinite(rho) else None)
    tried = np.zeros(len(idx), bool)
    ctx, first, updates = [], 0, 0
    for attempt in range(1, 101):
        cand = np.where(tried, -np.inf, sc)
        j = int(np.argmax(cand))
        assert not tried[j]
        tried[j] = True
        attempts_trace.append({'attempt': attempt, 'candidate_local': j, 'pack_index': int(idx[j]),
                               'action': A_[idx[j]].tolist(), 'context': list(ctx),
                               'score': float(sc[j]), 'solved': bool(Y_[idx[j]]),
                               'score_snapshot': len(score_history) - 1})
        if Y_[idx[j]]:
            first = attempt
            break
        if args.feedback and mem_model is not None and updates < args.max_updates:
            ctx = (ctx + [idx[j]])[-args.keep:]
            sc = score_all(idx, ctx)
            updates += 1
        if args.tta_steps and updates < args.max_updates:
            fails.append(idx[j])
            z = torch.from_numpy(np.stack([lat[i] for i in fails])).to(dev).float()
            a_ = torch.from_numpy(A_[fails]).to(dev)
            tmodel.train()
            opt = torch.optim.Adam(tmodel.parameters(), args.tta_lr)
            with torch.enable_grad():
                for _ in range(args.tta_steps):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = (tmodel(z[:, 0], a_).float()[:, 1:] - z[:, 1:]).abs().mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            tmodel.eval()
            sc = score_all(idx, [], tmodel)
            updates += 1
    per.append({'task': task, 'first_success': first, 'n_solutions_in_shortlist': int(Y_[idx].sum()), 'updates': updates,
                'auccess': float((WEIGHTS * (np.arange(1, 101) >= first)).sum() / WEIGHTS.sum()) if first else 0.0})
    score_path = parts / (task.replace(':', '_') + '_scores.npz')
    np.savez_compressed(score_path, scores=np.stack(score_history), indices=np.array(idx), labels=Y_[idx])
    save_json(part, {'result': per[-1], 'initial_rho': rhos[-1], 'args': vars(args),
                     'provenance': provenance, 'attempts': attempts_trace, 'contexts': context_history,
                     'score_channels': ['legacy_tanh32', 'stable_log64'], 'score_file': score_path.name})
    if ti % 10 == 0:
        print(ti, task, 'first', first, 'updates', updates, f'{time.time() - t0:.0f}s', flush=True)
fr = np.array([r['first_success'] for r in per])
agg = {'memory': args.memory, 'feedback': args.feedback, 'n_tasks': len(per), 'auccess': round(float(np.mean([r['auccess'] for r in per])), 4),
       'success_at': {k: round(float(((fr > 0) & (fr <= k)).mean()), 3) for k in (1, 2, 3, 5, 10, 100)},
       'mean_spearman_initial_vs_base_rank': round(float(np.mean([r for r in rhos if r is not None])), 3) if any(r is not None for r in rhos) else None, 'seconds': round(time.time() - t0)}
save_json(args.out, {'aggregate': agg, 'per_task': per, 'args': vars(args), 'provenance': provenance, 'score_mode': args.score_mode})
print(json.dumps(agg, indent=1))
