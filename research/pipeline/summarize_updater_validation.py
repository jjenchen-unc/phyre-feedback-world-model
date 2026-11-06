"""Offline audit of the numerical-control experiments; no relaxed acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def first_difference(a,b):
    assert len(a)==len(b)
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)


def summarize(root):
    records={}
    names=['STRICT_CUDA_cross','STRICT_CUDA_within','STRICT_CPU_cross','DEFAULT_CUDA_cross']
    for name in names:
        path=root/(name+'.json');d=json.loads(path.read_text())
        assert 'progress' not in d and len(d['arms'])==3,(name,'unfinished')
        assert d['slots']==65 and d['random_slots']==64
        assert len(d['stress_tasks'])==2 and d['audited_scores']['scores']>0
        comparisons=[];base=d['arms'][0]
        for arm in d['arms']:
            assert len(arm['events'])==65
            assert arm['frozen_predictor_exact'] and arm['probe_exact']
            assert len(arm['events'][-1])==2
            assert all(e['used_updates']==8 for e in arm['events'][-1])
            assert len(arm['gradient_sha256'])==len(arm['weight_sha256'])==65
        for arm in d['arms'][1:]:
            row={'arm':arm['name']}
            for field in ['events','gradient_sha256','weight_sha256']:
                row['first_differing_'+field+'_slot']=first_difference(base[field],arm[field])
            row['first_differing_policy_slot']=first_difference(
                [[{k:e[k] for k in ['selected','supports','solved']} for e in slot] for slot in base['events']],
                [[{k:e[k] for k in ['selected','supports','solved']} for e in slot] for slot in arm['events']])
            row['exact']=all(row['first_differing_'+f+'_slot'] is None for f in ['events','gradient_sha256','weight_sha256'])
            comparisons.append(row)
        assert d['complete']==all(c['exact'] for c in comparisons)
        records[name]={'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'passed_exact':d['complete'],'comparisons':comparisons,
            'checkpoint_sha256':d['checkpoint_sha256'],'episode_sha256':d['episode_sha256'],
            'strict_algorithms':d['strict_algorithms'],'pool_backward':d['pool_backward'],
            'audited_scores':d['audited_scores'],'pool_shapes':d['pool_shapes'],
            'pool_backward_calls':d['pool_backward_calls'],'slots_per_arm':65,'episodes_per_arm':130,
            'max_updates':max(e['used_updates'] for a in d['arms'] for slot in a['events'] for e in slot)}
    cross=[records[k] for k in names if k.endswith('cross')]
    assert len({r['checkpoint_sha256'] for r in cross})==1
    assert len({r['episode_sha256'] for r in cross})==1
    out={'audit_complete':True,'strict_acceptance':'all event/gradient/weight hashes exact; no tolerance relaxation',
         'shared_gpu_timing':True,'formal_training_modified':False,'runs':records}
    (root/'SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();summarize(a.root)
