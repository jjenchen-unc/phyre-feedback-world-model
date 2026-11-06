import argparse, glob, json, os, time
from multiprocessing import Pool
import numpy as np
import phyre
from phyre import simulator as S

p = argparse.ArgumentParser()
p.add_argument('--candidates', default='/mnt/meg/jjenchen/phyre-unc/candidates/ball_within_template_f0_dev_n100')
p.add_argument('--rule', nargs=3, default=['soft', '4', '1'])
p.add_argument('--workers', type=int, default=32)
p.add_argument('--out', required=True)
args = p.parse_args()
RULE = (args.rule[0], float(args.rule[1]), float(args.rule[2]))
W = 256
GREEN, BLUE, PURPLE = 9, 10, 11
GF = list(range(0, 128, 4))
STEPS = [g * 8 for g in GF]
cache = phyre.get_default_100k_cache('ball')
ACT = cache.action_array


def surface_dist(f):
    g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
    goals = np.flatnonzero((f[0, :, BLUE] > .5) | (f[0, :, PURPLE] > .5))
    cg, rg = f[:, g, :2] * W, f[0, g, 3] * W / 2
    ds = []
    for o in goals:
        co, L = f[:, o, :2] * W, f[0, o, 3] * W
        if np.argmax(f[0, o, 4:8]) == 1:
            th = f[:, o, 2] * 2 * np.pi
            u = np.stack([np.cos(th), np.sin(th)], -1)
            t = np.clip(((cg - co) * u).sum(-1), -L / 2, L / 2)
            ds.append(np.linalg.norm(cg - (co + t[:, None] * u), axis=-1) - rg)
        else:
            ds.append(np.linalg.norm(cg - co, axis=-1) - rg - L / 2)
    return np.min(ds, 0)


def longest(m):
    b = r = 0
    for v in m:
        r = r + 1 if v else 0
        b = max(b, r)
    return b


def score(d):
    kind, tau, s = RULE
    return longest(d < tau) if kind == 'hard' else float((0.5 * (1 + np.tanh((tau - d) / (2 * s)))).sum())


def work(path):
    task = os.path.basename(path)[:-4].replace('_', ':')
    z = np.load(path)
    sim = phyre.initialize_simulator([task], 'ball')
    out = []
    for i in z['action_idx']:
        ui, ok = sim._action_mapper.action_to_user_input(ACT[i])
        scenes = S.simulate_scene(S.add_user_input_to_scene(sim._tasks[0].scene, ui, sim._keep_spaces), 1017)
        f = np.concatenate([S.scene_to_featurized_objects(scenes[k]).features for k in STEPS])
        out.append(score(surface_dist(f)))
    return path, np.array(out), z['status']


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.candidates, '*.npz')))
    with Pool(args.workers) as pool:
        for k, (path, sc, st) in enumerate(pool.imap_unordered(work, files)):
            np.savez(os.path.join(args.out, os.path.basename(path)), scores=sc, status=st)
            if k % 10 == 0:
                print(k, f'{time.time() - t0:.0f}s', flush=True)
    print('done', f'{time.time() - t0:.0f}s')
