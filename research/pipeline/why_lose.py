import glob, os
import numpy as np
M = '/mnt/meg/jjenchen/phyre-unc'
rows = []
for f in sorted(glob.glob(f'{M}/cl_scores_base_s16r16/*.npz')):
    b = os.path.basename(f)
    wm, clf, tr = np.load(f), np.load(f'{M}/cl_scores_sceneclf/{b}'), np.load(f'{M}/cl_truth_rule_within_f0_dev/{b}')
    st = wm['status'] > 0
    truth = tr['scores']
    for name, sc in [('predictor', wm['scores']), ('initial_scene', clf['scores'])]:
        top = np.argsort(-sc, kind='stable')[:10]
        fails = top[~st[top]]
        rows.append((name, b[:5], len(fails), (truth[fails] > 2).mean() if len(fails) else np.nan,
                     (truth[~st] > 2).mean(), np.mean(st[top])))
for name in ['predictor', 'initial_scene']:
    r = [x for x in rows if x[0] == name]
    print(f'{name:14} top-10 precision {np.mean([x[5] for x in r]):.3f} | failures in top-10: {np.mean([x[2] for x in r]):.1f}/10 | '
          f'of those, near-contact by true trajectory (truth score>2): {np.nanmean([x[3] for x in r]):.2f} vs base rate among all failures {np.mean([x[4] for x in r]):.2f}')
tpl = sorted({x[1] for x in rows})
print('per-template top-10 precision (predictor / initial scene):')
print(' '.join(f"{t}:{np.mean([x[5] for x in rows if x[0]=='predictor' and x[1]==t]):.2f}/{np.mean([x[5] for x in rows if x[0]=='initial_scene' and x[1]==t]):.2f}" for t in tpl))
