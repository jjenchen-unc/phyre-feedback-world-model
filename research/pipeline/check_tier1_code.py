import numpy as np, torch
from common import Predictor, AttnProbe
dev = torch.device('cuda')
W, G, GREEN, PURPLE, BLACK = 256, 14, 9, 11, 13
P = '/mnt/meg/jjenchen/phyre-unc/packs/'
pc = torch.load('/mnt/meg/jjenchen/phyre-unc/runs/base_s16r16_pca128/predictor.pt', map_location='cpu', weights_only=False); pa = pc['args']
pred = Predictor(pc['Din'], pc['T'], pc['S'], pa['d'], pa['layers'], pa['heads']).to(dev); pred.load_state_dict(pc['state_dict']); pred.eval()
qc = torch.load('/path/to/phyre-unc/results/probes_pca128/ball_within_template_f0_full.pt', map_location='cpu', weights_only=False)
probe = AttnProbe(qc['D'], qc['S']).to(dev); probe.load_state_dict(qc['state_dict']); probe.eval()
for name in ['within_f0_dev_clusters_pca128']:
    lat = np.load(P + name + '_latents.npy', mmap_mode='r'); m = np.load(P + name + '_meta.npz')
    F_, M_, A_ = m['features'], m['mask'], m['action']
    errs, errs_real = [], []
    with torch.no_grad():
        for i in range(0, len(lat), 16):
            z = torch.from_numpy(np.ascontiguousarray(lat[i:i + 16])).to(dev).float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pr = pred(z[:, 0], torch.from_numpy(A_[i:i + 16]).to(dev)).float()
            B, T = pr.shape[:2]
            for src, store in [(pr, errs), (z, errs_real)]:
                kv = probe.inp(src.reshape(B * T, G * G, -1)) + probe.space
                for b in range(B):
                    f = F_[i + b][:, M_[i + b]]
                    g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
                    q = torch.from_numpy(np.repeat(f[:1, g], T, 0)).to(dev)
                    pxy, _ = probe.read(kv[b * T:(b + 1) * T], q)
                    store.append(np.linalg.norm(pxy.cpu().numpy() / W - f[:, g, :2], axis=-1)[1:] * W)
    tp = np.array([str(t)[:5] for t in m['tasks']])
    per = {}
    for k, t in enumerate(tp):
        per.setdefault(t, []).append(np.median(errs[k]))
    print('per-template median of per-rollout medians (pred):', {t: round(float(np.median(v)), 1) for t, v in sorted(per.items())})
    print('median of per-rollout medians overall: %.2f' % np.median([np.median(x) for x in errs]))
    e, er = np.concatenate(errs), np.concatenate(errs_real)
    print(f'{name}: first {min(len(lat),400)} rollouts | green err on predicted latents median {np.median(e):.2f}px | on real latents {np.median(er):.2f}px')
