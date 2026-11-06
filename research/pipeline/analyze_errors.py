import argparse, json, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--preds', nargs='+', required=True)
p.add_argument('--data', default='/path/to/phyre-unc/data')
p.add_argument('--out', required=True)
args = p.parse_args()
W = 256
GREEN, BLUE, PURPLE = 9, 10, 11
GF = list(range(0, 128, 4))


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


def q(v):
    v = np.asarray(v, float)
    return {'n': int(len(v)), 'median': round(float(np.median(v)), 2), 'p90': round(float(np.percentile(v, 90)), 2)} if len(v) else {'n': 0}


out = {}
for path in args.preds:
    z = np.load(path, allow_pickle=True)
    name = os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)[:-4]
    E, D_true, SPD, TIDX, TPL, GOALDYN = [], [], [], [], [], []
    dist_err_at_contact, rows = [], []
    for k, (fpath, pred) in enumerate(zip(z['files'], z['pred'])):
        f = np.load(os.path.join(args.data, fpath))['features'][GF]
        g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
        e = np.linalg.norm((pred[:, g, :2] - f[:, g, :2]) * W, axis=-1)
        dt, dp = surface_dist(f), surface_dist(pred)
        spd = np.r_[0, np.linalg.norm(np.diff(f[:, g, :2] * W, axis=0), axis=-1)]
        E.append(e); D_true.append(dt); SPD.append(spd); TIDX.append(np.arange(len(GF)))
        TPL.append(np.full(len(GF), str(z['tasks'][k])[:5]))
        dist_err_at_contact += list((dp - dt)[dt < 4])
        rows.append({'solved': bool(z['solved'][k]), 'e2e': float(z['e2e_scores'][k]), 'ref': float(z['ref_scores'][k]),
                     'min_true': float(dt.min()), 'min_pred': float(dp.min()), 'contact_frames_true': int((dt < 4).sum()),
                     'contact_frames_pred': int((dp < 4).sum())})
    E, D_true, SPD, TIDX, TPL = map(np.concatenate, (E, D_true, SPD, TIDX, TPL))
    r = {'green_err_all': q(E)}
    r['by_true_surface_dist'] = {lab: q(E[m]) for lab, m in [('<4px (contact)', D_true < 4), ('4-20px', (D_true >= 4) & (D_true < 20)),
                                                             ('20-60px', (D_true >= 20) & (D_true < 60)), ('>=60px', D_true >= 60)]}
    r['by_speed_px_per_step'] = {lab: q(E[m]) for lab, m in [('<1 (at rest)', SPD < 1), ('1-5', (SPD >= 1) & (SPD < 5)),
                                                            ('5-15', (SPD >= 5) & (SPD < 15)), ('>=15', SPD >= 15)]}
    r['by_time'] = {f'{a}-{b}': q(E[(TIDX >= a) & (TIDX < b)]) for a, b in [(0, 4), (4, 12), (12, 20), (20, 32)]}
    r['by_template_median'] = {t: round(float(np.median(E[TPL == t])), 2) for t in sorted(set(TPL))}
    r['pred_minus_true_surface_dist_at_true_contact_frames'] = q(dist_err_at_contact) | {'mean': round(float(np.mean(dist_err_at_contact)), 2)}
    succ = [x for x in rows if x['solved']]; fail = [x for x in rows if not x['solved']]
    thr = np.median([x['e2e'] for x in fail])
    fn = [x for x in succ if x['e2e'] <= thr]
    thr2 = np.median([x['e2e'] for x in succ])
    fp = [x for x in fail if x['e2e'] >= thr2]
    r['false_negatives'] = {'definition': 'successes scored <= median failure score', 'n': len(fn), 'of': len(succ),
                            'min_true_dist': q([x['min_true'] for x in fn]), 'min_pred_dist': q([x['min_pred'] for x in fn]),
                            'true_contact_frames': q([x['contact_frames_true'] for x in fn]), 'pred_contact_frames': q([x['contact_frames_pred'] for x in fn])}
    r['false_positives'] = {'definition': 'failures scored >= median success score', 'n': len(fp), 'of': len(fail),
                            'min_true_dist': q([x['min_true'] for x in fp]), 'min_pred_dist': q([x['min_pred'] for x in fp]),
                            'frac_near_miss_true_min_lt_10px': round(float(np.mean([x['min_true'] < 10 for x in fp])), 3) if fp else None}
    r['all_failures_frac_near_miss_lt_10px'] = round(float(np.mean([x['min_true'] < 10 for x in fail])), 3)
    out[name] = r
    print('==', name); print(json.dumps(r, indent=1))
json.dump(out, open(args.out, 'w'), indent=1)
