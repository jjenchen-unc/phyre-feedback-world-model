import glob, json
import numpy as np
R = '/path/to/phyre-unc/results/'
W = np.log(np.arange(2, 102)) - np.log(np.arange(1, 101))
for n in ['base', 'v1c_nofb', 'v1c_fb', 'v3_nofb', 'v3_fb', 'tta20', 'dqn_fixed', 'dqn_online', 'dqnm_fixed', 'dqnm_online']:
    files = [R + f'clm_{n}.json'] + sorted(glob.glob(R + f'x3_clm_{n}.json') + glob.glob(R + f'x3_clm_{n}_s*.json'))
    per = [r for f in files for r in json.load(open(f))['per_task']]
    tasks = [r['task'] for r in per]
    assert len(tasks) == len(set(tasks)), n
    fr = np.array([r['first_success'] for r in per])
    agg = {'n_tasks': len(per), 'files': files, 'auccess': round(float(np.mean([r['auccess'] for r in per])), 4),
           'success_at': {str(k): round(float(((fr > 0) & (fr <= k)).mean()), 3) for k in (1, 2, 3, 5, 10, 100)}}
    json.dump({'aggregate': agg, 'per_task': per}, open(R + f'w400_{n}.json', 'w'), indent=1)
    print(n, agg['n_tasks'], agg['auccess'])
