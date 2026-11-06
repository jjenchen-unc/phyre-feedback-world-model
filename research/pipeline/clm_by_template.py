import json, sys
import numpy as np
R = {n: {r['task']: r for r in json.load(open(f'/path/to/phyre-unc/results/{n}.json'))['per_task']} for n in sys.argv[1:]}
first = next(iter(R.values()))
for t5 in sorted({t[:5] for t in first}):
    ks = [k for k in first if k[:5] == t5]
    print(t5, '  '.join(f"{n}: {np.mean([R[n][k]['auccess'] for k in ks]):.2f}" for n in R))
