import numpy as np
from sklearn.metrics import roc_auc_score
exec(open('pipeline/render_fn.py').read().split("z = np.load")[0])
GF = list(range(0, 128, 4))


def longest(m):
    b = r = 0
    for v in m:
        r = r + 1 if v else 0
        b = max(b, r)
    return b


z = np.load('results/preds_levjepa_full_div16/ball_cross_template_full.npz', allow_pickle=True)
y, tasks = z['solved'], z['tasks']
tpl = np.array([t[:5] for t in tasks])
Dp = [surface_dist(p) for p in z['pred']]
Dt = [surface_dist(np.load('data/' + str(f))['features'][GF]) for f in z['files']]
print('rule                 | all   ' + ' '.join(f'{t:>6}' for t in sorted(set(tpl))) + ' | true-state all')
for kind, tau, s in [('soft', 2, 1), ('soft', 4, 2), ('soft', 8, 4), ('soft', 12, 4), ('hard', 3, 0), ('hard', 6, 0), ('hard', 10, 0), ('hard', 16, 0)]:
    f = (lambda D: np.array([longest(d < tau) for d in D])) if kind == 'hard' else (lambda D: np.array([(0.5 * (1 + np.tanh((tau - d) / (2 * s)))).sum() for d in D]))
    sp, st = f(Dp), f(Dt)
    per = ' '.join(f'{roc_auc_score(y[tpl == t], sp[tpl == t]):6.3f}' for t in sorted(set(tpl)))
    print(f'{kind:4} tau={tau:<3} s={s:<2}     | {roc_auc_score(y, sp):.3f} {per} | {roc_auc_score(y, st):.3f}')
m = tpl == '00011'
fn = [(round(float(Dt[k].min()), 1), round(float(Dp[k].min()), 1), int((Dt[k] < 4).sum()), int((Dp[k] < 4).sum())) for k in np.flatnonzero(m & (y > 0))]
fp = [(round(float(Dt[k].min()), 1), round(float(Dp[k].min()), 1), int((Dt[k] < 4).sum()), int((Dp[k] < 4).sum())) for k in np.flatnonzero(m & (y == 0))]
print('00011 successes (true min, pred min, true contact frames, pred contact frames):', sorted(fn, key=lambda x: x[1])[-8:])
print('00011 failures:', sorted(fp, key=lambda x: x[1])[:8])
