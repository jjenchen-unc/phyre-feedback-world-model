import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
eps = json.load(open('/path/to/phyre-unc/results/tier1_episodes_within_f0_dev.json'))
a, b = json.load(open(sys.argv[1]))['per_episode_query'], json.load(open(sys.argv[2]))['per_episode_query']
m = np.load('/mnt/meg/jjenchen/phyre-unc/packs/within_f0_dev_clusters_pca128_meta.npz')
Y = m['solved'].astype(int)
names = ['latent_l1', 'object_latent_l1', 'green_err_px']
for kind in ['k1', 'k2', 'k4', 'scorer_fail']:
    for split in ['all', 'near', 'far']:
        d = {m: [] for m in range(3)}
        tasks = []
        auc_d, auc_t = [], []
        for ei, e in enumerate(eps):
            if e['kind'] != kind:
                continue
            qs = e['queries'] if split == 'all' else (e['near'] if split == 'near' else [q for q in e['queries'] if q not in e['near']])
            lab = [Y[q] for q in qs]
            if 0 < sum(lab) < len(lab):
                auc_d.append(roc_auc_score(lab, [b[str(ei)][str(q)][3] for q in qs]) - roc_auc_score(lab, [a[str(ei)][str(q)][3] for q in qs]))
                auc_t.append(e['task'])
            for q in qs:
                x, y = a[str(ei)][str(q)], b[str(ei)][str(q)]
                for m in range(3):
                    d[m].append(y[m] - x[m])
                tasks.append(e['task'])
        if not tasks:
            continue
        tasks = np.array(tasks)
        row = f'{kind:11} {split:4} n={len(tasks):5}'
        for m in range(3):
            v = np.array(d[m])
            ok = ~np.isnan(v)
            per_task = np.array([v[ok & (tasks == t)].mean() for t in np.unique(tasks[ok])])
            se = per_task.std(ddof=1) / np.sqrt(len(per_task))
            row += f' | {names[m]}: {v[ok].mean():+.5f} (task-level SE {se:.5f}, z={per_task.mean() / se:+.1f})'
        if auc_d:
            auc_d, auc_t = np.array(auc_d), np.array(auc_t)
            pt = np.array([auc_d[auc_t == t].mean() for t in np.unique(auc_t)])
            se = pt.std(ddof=1) / np.sqrt(len(pt))
            row += f' | rank_auc: {auc_d.mean():+.4f} (SE {se:.4f}, z={pt.mean() / se:+.1f})'
        print(row)
