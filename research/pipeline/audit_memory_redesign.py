"""Audit the frozen merged dataset and paired episode draws before training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from redesign_lib import Store, write_json, digest
from redesign_sampling import sample_episode


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();data=json.loads(Path(a.data).read_text());store=Store(include_features=False)
    assert data['files_sha256']==hashlib.sha256('\n'.join(store.files).encode()).hexdigest()
    assert not set(data['train_tasks']) & set(data['validation_tasks'])
    assert not {t[:5] for t in data['train_tasks']} & {t[:5] for t in data['validation_tasks']}
    for t,d in data['tasks'].items():
        assert not set(d['support_pool']) & set(d['query_pool'])
        assert all(store.tasks[i]==t for i in d['support_pool']+d['query_pool'])
        for name in ['base_trace','memory_trace']:
            tr=d[name];attempts=tr['attempts']
            assert not any(r['solved'] for r in attempts[:-1])
            assert [r['index'] for r in attempts if not r['solved']]==tr['failures']
            assert all(r['solved']==bool(store.labels[r['index']]) for r in attempts)
    counts={v:Counter() for v in ['R0','R1','R2','R3','R4']}
    for step in range(2000):
        es={v:sample_episode(data,store,step,0,v) for v in counts}
        assert len({e['task'] for e in es.values()})==1
        assert es['R0']['query']==es['R1']['query']
        assert es['R2']['query']==es['R3']['query']==es['R4']['query']
        assert es['R0']['context']==es['R2']['context']
        assert es['R1']['context']==es['R3']['context']==es['R4']['context']
        for v,e in es.items():
            c=counts[v];c['episodes']+=1;c['queries']+=len(e['query'])
            c['positive_queries']+=int(store.labels[e['query']].sum())
            c['source_'+e['context_kind']]+=1;c['actual_k_'+str(len(e['context']))]+=1
            c['fallbacks']+=int(e['context_fallback'] is not None)
            if v in ['R1','R3','R4']:assert not store.labels[e['context']].any()
            if e['context_kind'] in ['base_trace','memory_trace'] and not e['context_fallback']:
                assert e['context']==data['tasks'][e['task']][e['context_kind']]['failures'][:e['requested_k']]
    result={'passed':True,'draws_per_variant':2000,'counts':{v:dict(c) for v,c in counts.items()},
            'sampler_sha256':digest(Path(__file__).with_name('redesign_sampling.py')),
            'data_sha256':digest(a.data),'train_tasks':len(data['train_tasks']),
            'validation_tasks':len(data['validation_tasks'])}
    write_json(a.out,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
