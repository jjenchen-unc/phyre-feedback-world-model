import numpy as np
from collections import defaultdict
m = np.load('/mnt/meg/jjenchen/phyre-unc/packs/within_f0_train_s16r16_pca128_meta.npz')
A, F, M, Y, T = m['action'], m['features'], m['mask'], m['solved'], m['tasks']
W = 256
by = defaultdict(list)
for i, t in enumerate(T):
    by[t].append(i)


def moved_set(i):
    f, mk = F[i], M[i]
    dyn = [o for o in np.flatnonzero(mk) if f[0, o, 13] < .5 and f[0, o, 11] < .5 and f[0, o, 8] < .5]
    return frozenset(o for o in dyn if np.linalg.norm((f[:, o, :2] - f[0, o, :2]) * W, axis=-1).max() > 5)


nn, pairs_close, pairs_close_diffout, pairs_sig, pairs_sig_diffout, pairs_sig_close_diffout, n_pairs = [], 0, 0, 0, 0, 0, 0
for t, idx in by.items():
    idx = np.array(idx)
    a = A[idx]
    d = np.linalg.norm(a[:, None] - a[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn += list(d.min(1))
    sig = [moved_set(i) for i in idx]
    y = Y[idx]
    for p in range(len(idx)):
        for q in range(p + 1, len(idx)):
            n_pairs += 1
            close = d[p, q] < 0.1
            same = sig[p] == sig[q]
            diff = y[p] != y[q]
            pairs_close += close; pairs_close_diffout += close and diff
            pairs_sig += same; pairs_sig_diffout += same and diff
            pairs_sig_close_diffout += same and diff and close
nn = np.array(nn)
print('tasks', len(by), 'rollouts', len(A), 'within-task pairs', n_pairs)
print('nearest-neighbour action distance (x,y,r in [0,1]): median %.3f p10 %.3f p90 %.3f; frac < 0.05: %.3f; < 0.1: %.3f' % (np.median(nn), np.percentile(nn, 10), np.percentile(nn, 90), (nn < .05).mean(), (nn < .1).mean()))
print('pairs with action distance < 0.1: %d (%.2f%%), of which different outcome: %d' % (pairs_close, 100 * pairs_close / n_pairs, pairs_close_diffout))
print('pairs with same moved-object set: %d (%.1f%%), different outcome: %d; + distance<0.1: %d' % (pairs_sig, 100 * pairs_sig / n_pairs, pairs_sig_diffout, pairs_sig_close_diffout))
print('per task: mean pairs close&diff-outcome %.2f, tasks with >=1 such pair %.2f' % (pairs_close_diffout / len(by), np.mean([1])))
