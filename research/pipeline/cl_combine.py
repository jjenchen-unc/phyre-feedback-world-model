import argparse, glob, json, os
import numpy as np
from scipy.stats import rankdata

p = argparse.ArgumentParser()
p.add_argument('--clf', required=True)
p.add_argument('--wm', required=True)
p.add_argument('--k', type=int, default=512)
p.add_argument('--out', required=True)
args = p.parse_args()
W = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))


def metrics(order, status):
    solved = status[order] > 0
    first = int(np.argmax(solved)) + 1 if solved.any() else 0
    return (float((W * (np.arange(1, 101) >= first)).sum() / W.sum()) if first else 0.0), first


res = {m: [] for m in ['classifier', 'world_model', 'A_shortlist_rerank', 'B_rank_average', 'recall_at_k']}
for f in sorted(glob.glob(os.path.join(args.clf, '*.npz'))):
    c, w = np.load(f), np.load(os.path.join(args.wm, os.path.basename(f)))
    assert (c['status'] == w['status']).all()
    sc, sw, st = c['scores'], w['scores'], c['status']
    oc = np.argsort(-sc, kind='stable')
    res['classifier'].append(metrics(oc, st))
    res['world_model'].append(metrics(np.argsort(-sw, kind='stable'), st))
    top = oc[:args.k]
    res['A_shortlist_rerank'].append(metrics(top[np.lexsort((-sc[top], -sw[top]))], st))
    rank = rankdata(-sc) + rankdata(-sw)
    res['B_rank_average'].append(metrics(np.lexsort((-sc, rank)), st))
    res['recall_at_k'].append((float((st[top] > 0).any()), 0))
agg = {}
for m, v in res.items():
    a = np.array([x[0] for x in v]); fr = np.array([x[1] for x in v])
    agg[m] = {'auccess': round(float(a.mean()), 4), 'success@1': round(float((fr == 1).mean()), 3),
              'success@10': round(float(((fr > 0) & (fr <= 10)).mean()), 3)} if m != 'recall_at_k' else {'tasks_with_solution_in_topk': round(float(a.mean()), 3)}
agg['k'] = args.k; agg['n_tasks'] = len(res['classifier']); agg['clf'] = args.clf; agg['wm'] = args.wm
json.dump(agg, open(args.out, 'w'), indent=1)
print(json.dumps(agg, indent=1))
