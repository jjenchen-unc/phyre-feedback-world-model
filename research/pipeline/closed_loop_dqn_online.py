import argparse, copy, json, os, time
from collections import defaultdict
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModel
from common import SceneClassifier

p = argparse.ArgumentParser()
p.add_argument('--scenes', default='/mnt/meg/jjenchen/phyre-unc/scenes')
p.add_argument('--setup', default='ball_within_template_f0')
p.add_argument('--shortlist', default='/path/to/phyre-unc/data/shortlist_within_f0_dev_top512.jsonl')
p.add_argument('--classifier', required=True)
p.add_argument('--online', action='store_true')
p.add_argument('--steps', type=int, default=10)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--replay', type=int, default=32)
p.add_argument('--max-updates', type=int, default=10)
p.add_argument('--out', required=True)
args = p.parse_args()
dev = torch.device('cuda')
torch.manual_seed(0); rng = np.random.default_rng(0)
WEIGHTS = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))
WAD = torch.tensor([[255, 255, 255], [0xf3, 0x4f, 0x46], [0x6b, 0xce, 0xbb], [0x18, 0x77, 0xf2],
                    [0x4b, 0x4a, 0xa4], [0xb9, 0xca, 0xd2], [0, 0, 0], [0xfc, 0xdf, 0xe3]], dtype=torch.float32, device=dev) / 255
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def encode(codes, enc):
    out = []
    for i in range(0, len(codes), 32):
        x = WAD[torch.from_numpy(codes[i:i + 32].astype(np.int64)).to(dev)].flip(1).permute(0, 3, 1, 2)
        x = ((F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True) - MEAN) / STD).half()
        out.append(enc(pixel_values=torch.stack([x, x], 2)).last_hidden_state[:, 1:1 + 196].float())
    return torch.cat(out)


t0 = time.time()
enc = AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large', trust_remote_code=True, revision='e831a0347737fcaa660b39c57d41c109de399845').eval().to(dev).half()
dv = np.load(os.path.join(args.scenes, args.setup + '_dev.npz'))
tr = np.load(os.path.join(args.scenes, args.setup + '_train.npz'))
Zdv, Ztr = encode(dv['frames'], enc), encode(tr['frames'], enc)
del enc; torch.cuda.empty_cache()
ACT = torch.from_numpy(tr['actions']).to(dev)
st_tr = tr['status']
pos, neg = np.argwhere(st_tr > 0), np.argwhere(st_tr < 0)
dtask = {str(t): i for i, t in enumerate(dv['tasks'])}
base = SceneClassifier().to(dev)
base.load_state_dict(torch.load(args.classifier, map_location='cpu'))
base.eval()
by = defaultdict(list)
for l in open(args.shortlist):
    r = json.loads(l)
    by[r['task']].append((r['action_idx'], r['status']))


@torch.no_grad()
def scores(model, tok, a):
    return torch.cat([model(tok.expand(len(a[i:i + 512]), -1, -1), a[i:i + 512]) for i in range(0, len(a), 512)]).cpu().numpy()


per = []
for ti, task in enumerate(sorted(by)):
    acts = torch.from_numpy(tr['actions'][[x[0] for x in by[task]]]).to(dev)
    solved = np.array([x[1] > 0 for x in by[task]])
    tok = Zdv[dtask[task]][None]
    model = copy.deepcopy(base) if args.online else base
    sc = scores(model, tok, acts)
    tried = np.zeros(len(sc), bool); fails, first, updates = [], 0, 0
    for attempt in range(1, 101):
        j = int(np.argmax(np.where(tried, -np.inf, sc)))
        tried[j] = True
        if solved[j]:
            first = attempt
            break
        if args.online and updates < args.max_updates:
            fails.append(j)
            opt = torch.optim.Adam(model.parameters(), args.lr)
            model.train()
            for _ in range(args.steps):
                h = args.replay // 2
                rp, rn = pos[rng.integers(len(pos), size=h)], neg[rng.integers(len(neg), size=h)]
                rb = np.concatenate([rp, rn])
                tok_b = torch.cat([tok.expand(len(fails), -1, -1), Ztr[rb[:, 0]]])
                a_b = torch.cat([acts[fails], ACT[rb[:, 1]]])
                y_b = torch.cat([torch.zeros(len(fails)), torch.ones(h), torch.zeros(h)]).to(dev)
                loss = F.binary_cross_entropy_with_logits(model(tok_b, a_b), y_b)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            sc = scores(model, tok, acts)
            updates += 1
    per.append({'task': task, 'first_success': first, 'updates': updates,
                'auccess': float((WEIGHTS * (np.arange(1, 101) >= first)).sum() / WEIGHTS.sum()) if first else 0.0})
    if ti % 20 == 0:
        print(ti, task, 'first', first, f'{time.time() - t0:.0f}s', flush=True)
fr = np.array([r['first_success'] for r in per])
agg = {'classifier': args.classifier, 'online': args.online, 'n_tasks': len(per), 'auccess': round(float(np.mean([r['auccess'] for r in per])), 4),
       'success_at': {k: round(float(((fr > 0) & (fr <= k)).mean()), 3) for k in (1, 2, 3, 5, 10, 100)}, 'seconds': round(time.time() - t0)}
json.dump({'aggregate': agg, 'per_task': per, 'args': vars(args)}, open(args.out, 'w'), indent=1)
print(json.dumps(agg, indent=1))
