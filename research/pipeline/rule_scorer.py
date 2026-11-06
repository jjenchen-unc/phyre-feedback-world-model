import argparse, json, os
import numpy as np
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument('--data', default='/path/to/phyre-unc/data')
p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--setups', nargs='+', default=['ball_within_template', 'ball_cross_template'])
p.add_argument('--taus', type=float, nargs='+', default=[1, 2, 3, 4, 6, 8])
p.add_argument('--out', default='/path/to/phyre-unc/results/rule_scorer.json')
args = p.parse_args()
GREEN, BLUE, PURPLE, W = 9, 10, 11, 256
SHAPES = ['ball', 'bar', 'jar', 'sticks']


def surface_dist(f, g, o, geom):
    cg, rg = f[:, g, :2] * W, f[0, g, 3] * W / 2
    co, L = f[:, o, :2] * W, f[0, o, 3] * W
    shape = SHAPES[int(np.argmax(f[0, o, 4:8]))]
    if geom == 'segment' and shape == 'bar':
        th = f[:, o, 2] * 2 * np.pi
        u = np.stack([np.cos(th), np.sin(th)], -1)
        t = np.clip(((cg - co) * u).sum(-1), -L / 2, L / 2)
        return np.linalg.norm(cg - (co + t[:, None] * u), axis=-1) - rg
    return np.linalg.norm(cg - co, axis=-1) - rg - L / 2


def longest(mask):
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def features(f, geom):
    g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
    goals = np.flatnonzero((f[0, :, BLUE] > .5) | (f[0, :, PURPLE] > .5))
    d = np.min([surface_dist(f, g, o, geom) for o in goals], axis=0)
    return d, SHAPES[int(np.argmax(f[0, goals[0], 4:8]))]


def load(name, geom):
    rows = [json.loads(l) for l in open(os.path.join(args.data, 'manifests', name + '.jsonl'))]
    D, S, y = [], [], []
    for r in rows:
        d, s = features(np.load(os.path.join(args.data, r['file']))['features'], geom)
        D.append(d); S.append(s); y.append(r['solved'])
    return np.stack(D), np.array(S), np.array(y)


res = {}
for fold in args.folds:
    for setup in args.setups:
        key = f'{setup}_f{fold}'
        tr, dv = f'{setup}_f{fold}_train', f'{setup}_f{fold}_dev_n100'
        if not os.path.exists(os.path.join(args.data, 'manifests', dv + '.jsonl')):
            continue
        r = {}
        for geom in ['circle', 'segment']:
            Dtr, _, ytr = load(tr, geom)
            Ddv, Sdv, ydv = load(dv, geom)
            tau = max(args.taus, key=lambda t: roc_auc_score(ytr, [longest(d < t) for d in Dtr]))
            run = np.array([longest(d < tau) for d in Ddv])
            mind = -Ddv.min(1)
            by_shape = {s: round(float(roc_auc_score(ydv[Sdv == s], run[Sdv == s])), 3) for s in sorted(set(Sdv)) if len(set(ydv[Sdv == s])) > 1}
            r[geom] = {'tau_from_train': tau, 'auc_longest_contact_run': round(float(roc_auc_score(ydv, run)), 4),
                       'auc_neg_min_dist': round(float(roc_auc_score(ydv, mind)), 4),
                       'auc_run_by_goal_shape': by_shape, 'n_by_goal_shape': {s: int((Sdv == s).sum()) for s in sorted(set(Sdv))}}
        res[key] = r
        print(key, json.dumps(r), flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, 'w'), indent=1)
