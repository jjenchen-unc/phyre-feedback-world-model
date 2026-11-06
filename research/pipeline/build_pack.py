import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--latents', default='/mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32')
p.add_argument('--data', default='/path/to/phyre-unc/data')
p.add_argument('--manifests', nargs='+', required=True)
p.add_argument('--name', required=True)
p.add_argument('--out', default='/mnt/meg/jjenchen/phyre-unc/packs')
args = p.parse_args()
GF = list(range(0, 128, 4))
t0 = time.time()
rows = [json.loads(l) for m in args.manifests for l in open(os.path.join(args.data, 'manifests', m + '.jsonl'))]
os.makedirs(args.out, exist_ok=True)
first = np.load(os.path.join(args.latents, os.path.basename(rows[0]['file']).replace('.npz', '.npy')))
lat = np.lib.format.open_memmap(os.path.join(args.out, args.name + '_latents.npy'), mode='w+', dtype=np.float16, shape=(len(rows),) + first.shape)


def one(i):
    r = rows[i]
    lat[i] = np.load(os.path.join(args.latents, os.path.basename(r['file']).replace('.npz', '.npy')))
    z = np.load(os.path.join(args.data, r['file']))
    return i, z['features'][GF], z['action']


feats, acts = [None] * len(rows), [None] * len(rows)
with ThreadPoolExecutor(32) as ex:
    for k, (i, f, a) in enumerate(ex.map(one, range(len(rows)))):
        feats[i], acts[i] = f, a
        if k % 1000 == 0:
            print(k, f'{time.time() - t0:.0f}s', flush=True)
lat.flush()
nmax = max(f.shape[1] for f in feats)
F = np.zeros((len(rows), len(GF), nmax, 14), np.float32)
M = np.zeros((len(rows), nmax), bool)
for i, f in enumerate(feats):
    F[i, :, :f.shape[1]] = f
    M[i, :f.shape[1]] = True
np.savez(os.path.join(args.out, args.name + '_meta.npz'), features=F, mask=M, action=np.stack(acts).astype(np.float32),
         solved=np.array([r['solved'] for r in rows]), tasks=np.array([r['task'] for r in rows]), files=np.array([r['file'] for r in rows]))
print(json.dumps({'name': args.name, 'n': len(rows), 'shape': lat.shape, 'seconds': round(time.time() - t0)}))
