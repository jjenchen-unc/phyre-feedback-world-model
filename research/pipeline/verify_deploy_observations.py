"""Verify exact canonical first-frame fallback on two real failed guard cases."""
import sys
sys.path.insert(0,'/path/to/phyre-unc/results/deploy_pool_updater_20261016/code')
import deploy_pool_data as d
import numpy as np


def main():
    m=d.runtime().m;records=[]
    for panel,task,ai in [('cross','00008:197',3393),('within','00001:001',4232)]:
        pool,status,_=m.input_case(panel,task);pool['status']=status
        local=int(np.flatnonzero(pool['action_idx']==ai)[0]);cache=d.TruthCache(panel)
        try:observation=cache.get(task,local,pool)
        finally:cache.close()
        assert observation['action_idx']==ai and observation['status']==int(status[local])
        key=f'{task.replace(":","_")}_{ai:06d}.npz'
        path=next((d.R/'observations'/panel).glob('*/'+key))
        with np.load(path) as z:
            assert bool(z['canonical_initial_verified'])
            records.append({'panel':panel,'task':task,'action_idx':ai,'canonical_initial_exact':True,
                            'max':float(z['initial_maxdiff']),'mean':float(z['initial_meandiff'])})
    d.write(d.R/'preflight/canonical_observation_check.json',{'passed':True,'records':records})
    print('CANONICAL_INITIAL_PASS',records,flush=True)


if __name__=='__main__':main()
