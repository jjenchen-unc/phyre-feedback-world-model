import argparse, glob, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel

p = argparse.ArgumentParser()
p.add_argument('--scenes', default='/mnt/meg/jjenchen/phyre-unc/scenes')
p.add_argument('--setup', default='ball_within_template_f0')
p.add_argument('--candidates', default='/mnt/meg/jjenchen/phyre-unc/candidates/ball_within_template_f0_dev_n100')
p.add_argument('--data-mode', choices=['all', 'matched'], default='all')
p.add_argument('--matched-manifest', default='/path/to/phyre-unc/data/manifests/ball_within_template_f0_train_sub400_div16.jsonl')
p.add_argument('--updates', type=int, default=100000)
p.add_argument('--batch', type=int, default=64)
p.add_argument('--out', required=True)
p.add_argument('--save-scores', default='')
args = p.parse_args()
dev = torch.device('cuda')
torch.manual_seed(0); np.random.seed(0)
WAD = torch.tensor([[255, 255, 255], [0xf3, 0x4f, 0x46], [0x6b, 0xce, 0xbb], [0x18, 0x77, 0xf2],
                    [0x4b, 0x4a, 0xa4], [0xb9, 0xca, 0xd2], [0, 0, 0], [0xfc, 0xdf, 0xe3]], dtype=torch.float32, device=dev) / 255
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
WEIGHTS = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))


@torch.no_grad()
def encode(codes, enc):
    out = []
    for i in range(0, len(codes), 32):
        x = WAD[torch.from_numpy(codes[i:i + 32].astype(np.int64)).to(dev)].flip(1).permute(0, 3, 1, 2)
        x = ((F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True) - MEAN) / STD).half()
        out.append(enc(pixel_values=torch.stack([x, x], 2)).last_hidden_state[:, 1:1 + 196].float().half().cpu())
    return torch.cat(out)


class SceneClassifier(nn.Module):
    def __init__(self, D=1024, S=196, d=256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.space = nn.Parameter(torch.zeros(1, S, d))
        self.act = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.body = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), 2)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Linear(d, 1)

    def forward(self, tok, a):
        h = self.inp(tok.float()) + self.space + self.act(a)[:, None]
        h = self.body(h)
        return self.out(self.pool(self.q.expand(len(h), -1, -1), h, h)[0][:, 0]).squeeze(-1)


t0 = time.time()
enc = AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large', trust_remote_code=True, revision='e831a0347737fcaa660b39c57d41c109de399845').eval().to(dev).half()
tr = np.load(os.path.join(args.scenes, args.setup + '_train.npz'))
dv = np.load(os.path.join(args.scenes, args.setup + '_dev.npz'))
Ztr, Zdv = encode(tr['frames'], enc), encode(dv['frames'], enc)
del enc
torch.cuda.empty_cache()
ACT = torch.from_numpy(tr['actions'])
st = torch.from_numpy(tr['status'].astype(np.int64))
if args.data_mode == 'all':
    pos = torch.nonzero(st > 0)
    neg = torch.nonzero(st < 0)
else:
    tix = {t: i for i, t in enumerate(tr['tasks'])}
    rows = [json.loads(l) for l in open(args.matched_manifest)]
    pairs = torch.tensor([(tix[r['task']], r['action_idx']) for r in rows])
    lab = torch.tensor([r['solved'] for r in rows])
    pos, neg = pairs[lab], pairs[~lab]
print(f'encoded in {time.time() - t0:.0f}s; train pos {len(pos)} neg {len(neg)}', flush=True)
model = SceneClassifier().to(dev)
opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.updates)
Ztr_d = Ztr.to(dev)
for step in range(args.updates):
    h = args.batch // 2
    bp, bn = pos[torch.randint(len(pos), (h,))], neg[torch.randint(len(neg), (h,))]
    b = torch.cat([bp, bn])
    y = torch.cat([torch.ones(h), torch.zeros(h)]).to(dev)
    logit = model(Ztr_d[b[:, 0]], ACT[b[:, 1]].to(dev))
    loss = F.binary_cross_entropy_with_logits(logit, y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if (step + 1) % 10000 == 0:
        print(f'step {step + 1} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
model.eval()
dtasks = list(dv['tasks'])
per = []
with torch.no_grad():
    for path in sorted(glob.glob(os.path.join(args.candidates, '*.npz'))):
        z = np.load(path)
        task = os.path.basename(path)[:-4].replace('_', ':')
        tok = Zdv[dtasks.index(task)][None].to(dev)
        a = torch.from_numpy(z['action']).to(dev)
        sc = torch.cat([model(tok.expand(len(a[i:i + 512]), -1, -1), a[i:i + 512]) for i in range(0, len(a), 512)]).cpu().numpy()
        if args.save_scores:
            os.makedirs(args.save_scores, exist_ok=True)
            np.savez(os.path.join(args.save_scores, os.path.basename(path)), scores=sc, status=z['status'])
        solved = z['status'][np.argsort(-sc, kind='stable')] > 0
        first = int(np.argmax(solved)) + 1 if solved.any() else 0
        per.append({'task': task, 'first': first, 'auccess': float((WEIGHTS * (np.arange(1, 101) >= first)).sum() / WEIGHTS.sum()) if first else 0.0})
agg = {'data_mode': args.data_mode, 'n_train_pos': len(pos), 'n_train_neg': len(neg), 'updates': args.updates,
       'auccess': round(float(np.mean([r['auccess'] for r in per])), 4),
       'success_at_1': round(float(np.mean([r['first'] == 1 for r in per])), 3),
       'success_at_10': round(float(np.mean([0 < r['first'] <= 10 for r in per])), 3), 'n_tasks': len(per), 'seconds': round(time.time() - t0)}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({'aggregate': agg, 'per_task': per}, open(args.out, 'w'), indent=1)
torch.save(model.state_dict(), args.out.replace('.json', '.pt'))
print(json.dumps(agg))
