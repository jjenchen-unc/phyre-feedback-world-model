import argparse, json, os
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--tasks', type=int, required=True)
p.add_argument('--data', default='/path/to/phyre-unc/data')
args = p.parse_args()
src = os.path.join(args.data, 'manifests', args.manifest + '.jsonl')
rows = [json.loads(l) for l in open(src)]
by_t = defaultdict(list)
for t in sorted({r['task'] for r in rows}):
    by_t[t.split(':')[0]].append(t)
pick, i = [], 0
while len(pick) < args.tasks and any(by_t.values()):
    for tp in sorted(by_t):
        if by_t[tp] and len(pick) < args.tasks:
            pick.append(by_t[tp].pop(0))
keep = set(pick)
out = os.path.join(args.data, 'manifests', f'{args.manifest}_sub{args.tasks}.jsonl')
with open(out, 'w') as f:
    for r in rows:
        if r['task'] in keep:
            f.write(json.dumps(r) + '\n')
print(out, len(keep), sum(r['task'] in keep for r in rows))
