import argparse, glob, json, os, shutil, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--packs', default='/mnt/meg/jjenchen/phyre-unc/packs')
p.add_argument('--fit-pack', default='within_f0_train_div16')
p.add_argument('--apply-packs', nargs='+', default=['within_f0_train_div16', 'within_f0_dev'])
p.add_argument('--ks', type=int, nargs='+', default=[256, 128])
p.add_argument('--n-fit-tokens', type=int, default=200000)
p.add_argument('--src-meta-dir', default='/mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32')
p.add_argument('--out', default='/mnt/meg/jjenchen/phyre-unc/latents')
args = p.parse_args()
dev = torch.device('cuda')
t0 = time.time()
lat = np.load(os.path.join(args.packs, args.fit_pack + '_latents.npy'), mmap_mode='r')
rng = np.random.default_rng(0)
N, T, S, D = lat.shape
ri = np.sort(rng.choice(N, 2000, replace=False))
X = torch.from_numpy(np.ascontiguousarray(lat[ri]).reshape(-1, D)).float()
X = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:args.n_fit_tokens]].to(dev)
mu = X.mean(0)
C = torch.cov((X - mu).T)
evals, evecs = torch.linalg.eigh(C)
order = torch.argsort(evals, descending=True)
evals, evecs = evals[order], evecs[:, order]
frac = (evals.cumsum(0) / evals.sum()).cpu().numpy()
info = {'explained_variance': {int(k): round(float(frac[k - 1]), 4) for k in [32, 64, 128, 256, 512]}}
print(info, flush=True)
for k in args.ks:
    odir = os.path.join(args.out, f'levjepa_overlap_g32_pca{k}')
    os.makedirs(odir, exist_ok=True)
    torch.save({'mean': mu.cpu(), 'components': evecs[:, :k].cpu(), 'explained': float(frac[k - 1])}, os.path.join(odir, 'pca.pt'))
    for m in glob.glob(os.path.join(args.src_meta_dir, 'meta_shard*.json')):
        meta = json.load(open(m)); meta['pca_k'] = k
        json.dump(meta, open(os.path.join(odir, os.path.basename(m)), 'w'))
W = {k: evecs[:, :k] for k in args.ks}
for name in args.apply_packs:
    L = np.load(os.path.join(args.packs, name + '_latents.npy'), mmap_mode='r')
    files = np.load(os.path.join(args.packs, name + '_meta.npz'))['files']
    for i in range(0, len(L), 64):
        x = torch.from_numpy(np.ascontiguousarray(L[i:i + 64])).to(dev).float() - mu
        for k in args.ks:
            y = (x @ W[k]).half().cpu().numpy()
            odir = os.path.join(args.out, f'levjepa_overlap_g32_pca{k}')
            for j, f in enumerate(files[i:i + 64]):
                np.save(os.path.join(odir, os.path.basename(str(f)).replace('.npz', '.npy')), y[j])
    print(name, 'projected', f'{time.time() - t0:.0f}s', flush=True)
json.dump(info, open(os.path.join(args.out, 'pca_info.json'), 'w'))
