import numpy as np
W = 256
GF = list(range(0, 128, 4))
GREEN, BLUE, PURPLE = 9, 10, 11
SH = ['ball', 'bar', 'jar', 'sticks']
for tag in ['preds_levjepa_full', 'preds_levjepa_full_div16']:
    z = np.load(f'results/{tag}/ball_cross_template_full.npz', allow_pickle=True)
    stats = {}
    for k in range(len(z['files'])):
        f = np.load('data/' + str(z['files'][k]))['features'][GF]
        p = z['pred'][k]
        tpl = str(z['tasks'][k])[:5]
        for o in np.flatnonzero(f[0, :, BLUE] > .5):
            e = np.linalg.norm((p[:, o, :2] - f[:, o, :2]) * W, axis=-1)
            key = (tpl, 'blue-' + SH[int(np.argmax(f[0, o, 4:8]))], round(float(f[0, o, 3] * W)))
            stats.setdefault(key, []).append(e)
        g = int(np.flatnonzero(f[0, :, GREEN] > .5)[0])
        stats.setdefault((tpl, 'green', 0), []).append(np.linalg.norm((p[:, g, :2] - f[:, g, :2]) * W, axis=-1))
    print('==', tag)
    for key in sorted(stats):
        e = np.concatenate(stats[key])
        sizes = sorted({key[2]})
        print('  ', key[0], key[1], 'n_rollouts', len(stats[key]), 'median %.1f p90 %.1f px' % (np.median(e), np.percentile(e, 90)))
f = None
import json
for setup in ['ball_cross_template_f0_train_sub400', 'ball_cross_template_f0_dev_n100']:
    rows = [json.loads(l) for l in open(f'data/manifests/{setup}.jsonl')]
    seen = {}
    for r in rows[::4]:
        ff = np.load('data/' + r['file'])['features'][0]
        for o in np.flatnonzero(ff[:, BLUE] > .5):
            seen.setdefault(SH[int(np.argmax(ff[o, 4:8]))], set()).add(r['task'][:5])
    print(setup, 'blue (dynamic goal) shapes by template:', {k: sorted(v) for k, v in seen.items()})
