import argparse, glob, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel
from common import W, T, GREEN, BLUE, PURPLE, BLACK, Predictor, AttnProbe, surface_dist, rule_score

p = argparse.ArgumentParser()
p.add_argument('--candidates', default='/mnt/meg/jjenchen/phyre-unc/candidates/ball_within_template_f0_dev_n100')
p.add_argument('--predictor', required=True)
p.add_argument('--probe', default='/path/to/phyre-unc/results/probes/ball_within_template_f0_full.pt')
p.add_argument('--max-tasks', type=int, default=0)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--out', required=True)
p.add_argument('--save-scores', default='')
p.add_argument('--pca', default='')
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=1)
args = p.parse_args()
dev = torch.device('cuda')
WAD = torch.tensor([[255, 255, 255], [0xf3, 0x4f, 0x46], [0x6b, 0xce, 0xbb], [0x18, 0x77, 0xf2],
                    [0x4b, 0x4a, 0xa4], [0xb9, 0xca, 0xd2], [0, 0, 0], [0xfc, 0xdf, 0xe3]], dtype=torch.float32, device=dev) / 255
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


WEIGHTS = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))


def auccess_from_first(first):
    return float((WEIGHTS * (np.arange(1, 101) >= first)).sum() / WEIGHTS.sum()) if first else 0.0


def random_auccess(n, m):
    if m == 0:
        return 0.0
    k = np.arange(1, 101)
    p_solved = 1 - np.array([np.prod([(n - m - i) / (n - i) for i in range(kk)]) if kk <= n - m else 0.0 for kk in k])
    return float((WEIGHTS * p_solved).sum() / WEIGHTS.sum())


enc = AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large', trust_remote_code=True, revision='e831a0347737fcaa660b39c57d41c109de399845').eval().to(dev).half()
pc = torch.load(args.predictor, map_location='cpu', weights_only=False)
pa = pc['args']
pred = Predictor(pc['Din'], pc['T'], pc['S'], pa['d'], pa['layers'], pa['heads']).to(dev)
pred.load_state_dict(pc['state_dict']); pred.eval()
qc = torch.load(args.probe, map_location='cpu', weights_only=False)
probe = AttnProbe(qc['D'], qc['S']).to(dev)
probe.load_state_dict(qc['state_dict']); probe.eval()
rule = qc['rule']
PCA = None
if args.pca:
    _p = torch.load(args.pca, map_location='cpu')
    PCA = (_p['mean'].to(dev), _p['components'].to(dev))

t0 = time.time()
files = sorted(glob.glob(os.path.join(args.candidates, '*.npz')))
files = files[:args.max_tasks] if args.max_tasks else files
files = files[args.shard::args.nshards]
per_task = []
for fi, path in enumerate(files):
    z = np.load(path)
    codes, feats0, act, status = z['frames'], z['feats0'], z['action'], z['status']
    dyn = [o for o in range(feats0.shape[1]) if feats0[0, o, BLACK] < .5 and feats0[0, o, PURPLE] < .5]
    scores = np.zeros(len(codes))
    with torch.no_grad():
        for i in range(0, len(codes), args.batch):
            x = WAD[torch.from_numpy(codes[i:i + args.batch].astype(np.int64)).to(dev)].flip(1).permute(0, 3, 1, 2)
            x = ((F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True) - MEAN) / STD).half()
            clip = torch.stack([x, x], 2)
            z0 = enc(pixel_values=clip).last_hidden_state[:, 1:1 + 196].float()
            if PCA:
                z0 = ((z0.half().float() - PCA[0]) @ PCA[1]).half().float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pr = pred(z0, torch.from_numpy(act[i:i + args.batch]).to(dev))
            B = len(z0)
            kv = probe.inp(pr.float().reshape(B * T, 196, -1)) + probe.space
            f = np.repeat(feats0[i:i + B, None], T, 1).copy()
            for o in dyn:
                q = torch.from_numpy(np.repeat(feats0[i:i + B, o], T, 0)).to(dev)
                pxy, pang = probe.read(kv, q)
                pxy = pxy.reshape(B, T, 2).cpu().numpy(); pang = pang.reshape(B, T, 2).cpu().numpy()
                f[:, :, o, :2] = pxy / W
                f[:, :, o, 2] = (np.arctan2(pang[..., 1], pang[..., 0]) / 2) / (2 * np.pi) % 1
            for b in range(B):
                scores[i + b] = rule_score(surface_dist(f[b]), rule)
    if args.save_scores:
        os.makedirs(args.save_scores, exist_ok=True)
        np.savez(os.path.join(args.save_scores, os.path.basename(path)), scores=scores, status=status, action=act)
    order = np.argsort(-scores, kind='stable')
    solved = status[order] > 0
    first = int(np.argmax(solved)) + 1 if solved.any() else 0
    n, m = len(status), int((status > 0).sum())
    rec = {'task': os.path.basename(path)[:-4].replace('_', ':'), 'n_candidates': n, 'n_solutions': m, 'first_success_rank': first,
           'auccess': auccess_from_first(first), 'random_auccess': random_auccess(n, m),
           'success_at': {k: bool(first and first <= k) for k in (1, 5, 10, 100)},
           'random_success_at': {k: 1 - float(np.prod([(n - m - i) / (n - i) for i in range(k)])) if m else 0.0 for k in (1, 5, 10, 100)}}
    per_task.append(rec)
    if fi % 10 == 0:
        print(fi, rec['task'], 'first', first, 'n_sol', m, f'{time.time() - t0:.0f}s', flush=True)
agg = {'n_tasks': len(per_task), 'auccess': round(float(np.mean([r['auccess'] for r in per_task])), 4),
       'random_auccess': round(float(np.mean([r['random_auccess'] for r in per_task])), 4),
       'success_at': {k: round(float(np.mean([r['success_at'][k] for r in per_task])), 3) for k in (1, 5, 10, 100)},
       'random_success_at': {k: round(float(np.mean([r['random_success_at'][k] for r in per_task])), 3) for k in (1, 5, 10, 100)},
       'tasks_with_no_solution_in_pool': sum(r['n_solutions'] == 0 for r in per_task),
       'predictor': args.predictor, 'probe': args.probe, 'rule': rule, 'seconds': round(time.time() - t0)}
json.dump({'aggregate': agg, 'per_task': per_task}, open(args.out, 'w'), indent=1)
print(json.dumps(agg, indent=1))
