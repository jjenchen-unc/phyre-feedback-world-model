import glob, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
W = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))


def auccess(scores, status):
    solved = status[np.argsort(-scores, kind='stable')] > 0
    first = int(np.argmax(solved)) + 1 if solved.any() else 0
    return float((W * (np.arange(1, 101) >= first)).sum() / W.sum()) if first else 0.0, first


out = {}
for name, d in [(n, p) for n, p in zip(sys.argv[1::2], sys.argv[2::2])]:
    per = {}
    for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
        z = np.load(f)
        a, first = auccess(z['scores'], z['status'])
        st = z['status'] > 0
        per[os.path.basename(f)[:-4]] = (a, first, float(roc_auc_score(st, z['scores'])) if 0 < st.sum() < len(st) else np.nan)
    tpl = sorted({k[:5] for k in per})
    out[name] = {'auccess': round(float(np.mean([v[0] for v in per.values()])), 4),
                 'success@1': round(float(np.mean([v[1] == 1 for v in per.values()])), 3),
                 'success@10': round(float(np.mean([0 < v[1] <= 10 for v in per.values()])), 3),
                 'pool_auc_mean': round(float(np.nanmean([v[2] for v in per.values()])), 3),
                 'by_template_auccess': {t: round(float(np.mean([v[0] for k, v in per.items() if k[:5] == t])), 2) for t in tpl}}
    print(name, json.dumps({k: v for k, v in out[name].items() if k != 'by_template_auccess'}))
names = list(out)
tpl = sorted(out[names[0]]['by_template_auccess'])
print('template ' + ' '.join(f'{n:>10}' for n in names))
for t in tpl:
    print(f'{t:8} ' + ' '.join(f"{out[n]['by_template_auccess'].get(t, float('nan')):10.2f}" for n in names))
