import json, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
from common import Predictor
dev = torch.device('cuda')
P = '/mnt/meg/jjenchen/phyre-unc/packs/within_f0_dev_clusters_pca128'
lat = np.load(P + '_latents.npy', mmap_mode='r'); meta = np.load(P + '_meta.npz')
A, T_, files = meta['action'], meta['tasks'], meta['files']
rows = {r['file']: r['cluster'] for r in map(json.loads, open('/path/to/phyre-unc/data/manifests/ball_within_template_f0_dev_clusters.jsonl'))}
C = np.array([rows[str(f)] for f in files])
pc = torch.load('/mnt/meg/jjenchen/phyre-unc/runs/base_s16r16_pca128/predictor.pt', map_location='cpu', weights_only=False); pa = pc['args']
pred = Predictor(pc['Din'], pc['T'], pc['S'], pa['d'], pa['layers'], pa['heads']).to(dev); pred.load_state_dict(pc['state_dict']); pred.eval()
R, Zd = [], []
with torch.no_grad():
    for i in range(0, len(lat), 32):
        z = torch.from_numpy(np.ascontiguousarray(lat[i:i + 32])).to(dev).float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pr = pred(z[:, 0], torch.from_numpy(A[i:i + 32]).to(dev)).float()
        def pool(x):
            B = x.shape[0]
            x = x[:, 4::4].reshape(B * 7, 14, 14, -1).permute(0, 3, 1, 2)
            return F.adaptive_avg_pool2d(x, 7).reshape(B, -1)
        R.append(pool(z - pr).cpu()); Zd.append(pool(z - z[:, :1]).cpu())
R, Zd = torch.cat(R), torch.cat(Zd)
Rn, Zn = F.normalize(R, dim=1), F.normalize(Zd, dim=1)
by = defaultdict(list)
for i, t in enumerate(T_):
    by[t].append(i)
near, far, near_z, far_z = [], [], [], []
for t, idx in by.items():
    for a in idx:
        for b in idx:
            if a < b:
                (near if C[a] == C[b] else far).append(float(Rn[a] @ Rn[b]))
                (near_z if C[a] == C[b] else far_z).append(float(Zn[a] @ Zn[b]))
rng = np.random.default_rng(0)
cross = [float(Rn[a] @ Rn[b]) for a, b in rng.integers(0, len(R), (20000, 2)) if T_[a] != T_[b]]
print('residual cosine (baseline error, true - predicted):  near %.3f | far %.3f | different tasks %.3f' % (np.mean(near), np.mean(far), np.mean(cross)))
print('true-motion cosine (z_t - z_0) for reference:        near %.3f | far %.3f' % (np.mean(near_z), np.mean(far_z)))
print('residual norm / motion norm (how big the error is vs the motion): %.3f' % float((R.norm(dim=1) / Zd.norm(dim=1)).mean()))
