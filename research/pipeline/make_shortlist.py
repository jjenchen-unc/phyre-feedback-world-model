import argparse, glob, json, os
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--candidates', required=True)
p.add_argument('--scores', required=True)
p.add_argument('--k', type=int, default=512)
p.add_argument('--exclude', default='')
p.add_argument('--out', required=True)
args = p.parse_args()
skip = {json.loads(l)['task'] for l in open(args.exclude)} if args.exclude else set()
rec, n = [], 0
with open(args.out, 'w') as out:
    for f in sorted(glob.glob(os.path.join(args.candidates, '*.npz'))):
        b = os.path.basename(f); task = b[:-4].replace('_', ':')
        if task in skip:
            continue
        c, s = np.load(f), np.load(os.path.join(args.scores, b))
        assert (c['status'] == s['status']).all()
        top = np.argsort(-s['scores'], kind='stable')[:args.k]
        for rank, i in enumerate(top):
            out.write(json.dumps({'task': task, 'action_idx': int(c['action_idx'][i]), 'status': int(c['status'][i]), 'base_rank': rank}) + '\n'); n += 1
        if (c['status'] > 0).any():
            rec.append(bool((c['status'][top] > 0).any()))
print(json.dumps({'rows': n, 'tasks': n // args.k, 'recall_at_k': round(float(np.mean(rec)), 3) if rec else None}))
