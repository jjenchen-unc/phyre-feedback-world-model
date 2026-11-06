import glob, json, sys
import numpy as np
R = '/path/to/phyre-unc/results/'
for n in sys.argv[1:]:
    per = [r for f in sorted(glob.glob(R + n + '_s[0-9].json')) for r in json.load(open(f))['per_task']]
    fr = np.array([r['first_success'] for r in per])
    agg = {'n_tasks': len(per), 'auccess': round(float(np.mean([r['auccess'] for r in per])), 4),
           'success_at': {str(k): round(float(((fr > 0) & (fr <= k)).mean()), 3) for k in (1, 2, 3, 5, 10, 100)}}
    json.dump({'aggregate': agg, 'per_task': per}, open(R + n + '.json', 'w'), indent=1)
    print(n, agg)
