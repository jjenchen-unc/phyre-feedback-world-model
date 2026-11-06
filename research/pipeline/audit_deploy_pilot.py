"""Build indexed pilot episodes and audit actual-K/query pairing before training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from deploy_pilot_lib import PilotStore,episode
from redesign_lib import write_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--old-data',required=True)
    a=p.parse_args();root=Path(a.root);sel=json.loads((root/'selection.json').read_text());old=json.loads(Path(a.old_data).read_text())
    store=PilotStore(include_features=False);index={str(f):i for i,f in enumerate(store.files)}
    assert hashlib.sha256('\n'.join(store.files[:store.old_n]).encode()).hexdigest()==old['files_sha256']
    data={'selection':sel,'tasks':{},'old_data_sha256':digest(a.old_data),'scope':sel['scope']};initial_errors=[]
    for task in sel['train_tasks']+sel['validation_tasks']:
        tr=json.loads((root/'traces'/f'{task.replace(":","_")}.json').read_text())
        idx={m['local']:index[m['file']] for m in tr['manifest']}
        score_path=Path('/mnt/meg/jjenchen/phyre-unc/deploy_pilot_20260929/scores')/f'{task.replace(":","_")}.npz'
        with np.load(score_path) as sc:
            orders={'base':np.argsort(-sc['scores'],kind='stable'),
                    'random':np.random.default_rng([20260929,*map(int,task.split(':'))]).permutation(len(sc['scores']))}
            for policy in ['base','random']:
                attempts=tr[policy]['attempts']
                assert [r['local'] for r in attempts]==orders[policy][:len(attempts)].tolist()
                assert all(r['action_idx']==int(sc['action_idx'][r['local']]) for r in attempts)
            # Audit encoder-route differences, without using future latents to
            # retroactively change the already executed collection policy.
            for j in list(idx)[:3]:
                delta=np.abs(store.z([idx[j]])[0,0].astype(float)-sc['initial_latent'][j].astype(float))
                initial_errors.append({'task':task,'local':j,'max_abs':float(delta.max()),'mean_abs':float(delta.mean())})
        for policy in ['base','random']:
            rows=tr[policy]['attempts'];assert not any(r['status']>0 for r in rows[:-1])
            assert [r['local'] for r in rows if r['status']<0]==tr[policy]['failures']
        es=[]
        for e in tr['episodes']:
            x={k:e[k] for k in ['attempt','actual_k']}
            for k in ['A','B','query']:x[k]=[idx[j] for j in e[k]]
            assert len(x['A'])==len(x['B'])==x['actual_k']>0
            assert not store.labels[x['A']+x['B']].any()
            assert not set(x['query']) & set(x['A']+x['B'])
            assert len(x['query'])==len(set(x['query']))>=8
            es.append(x)
        data['tasks'][task]={'episodes':es,'query_successes':tr['query_successes']}
    data['eligible_train_tasks']=[t for t in sel['train_tasks'] if data['tasks'][t]['episodes']]
    data['eligible_validation_tasks']=[t for t in sel['validation_tasks'] if data['tasks'][t]['episodes']]
    assert not {t[:5] for t in sel['train_tasks']} & {t[:5] for t in sel['validation_tasks']}
    data['files_sha256']=hashlib.sha256('\n'.join(store.files).encode()).hexdigest()
    # A small pilot must have enough actual nonempty paired evidence to train.
    data['ab_ready']=len(data['eligible_train_tasks'])>=8 and len(data['eligible_validation_tasks'])>=2
    counts=Counter()
    if data['ab_ready']:
        for seed in [0,1]:
            for step in range(2000):
                aa=episode(data,old,store,step,seed,'A');bb=episode(data,old,store,step,seed,'B')
                assert aa['task']==bb['task'] and aa['query']==bb['query'] and len(aa['context'])==len(bb['context'])
                if aa['source']=='legacy_R0':assert aa['context']==bb['context']
                assert not set(aa['query']) & set(aa['context']+bb['context'])
                counts[aa['source']]+=1;counts['actual_k_'+str(len(aa['context']))]+=1
    data['audit_counts']=dict(counts)
    write_json(root/'paired_data.json',data)
    write_json(root/'pairing_audit.json',{'passed':True,'ab_ready':data['ab_ready'],
        'eligible_train_scenes':len(data['eligible_train_tasks']),'eligible_validation_scenes':len(data['eligible_validation_tasks']),
        'matched_draws':sum(counts.get(x,0) for x in ['legacy_R0','deploy_pilot']),
        'data_sha256':digest(root/'paired_data.json'),'counts':dict(counts),
        'initial_encoding_comparison':{'samples':len(initial_errors),'max_abs':max(r['max_abs'] for r in initial_errors),
            'mean_abs':float(np.mean([r['mean_abs'] for r in initial_errors])),
            'scope':'deployment duplicated-initial-frame encoding vs recorded-trajectory first token; no policy re-selection'},
        'skip_reason':None if data['ab_ready'] else 'Fewer than 8 train or 2 validation scenes have honest nonempty paired prefixes; do not fabricate failures.'})
    print(json.dumps({'ab_ready':data['ab_ready'],'train':len(data['eligible_train_tasks']),'validation':len(data['eligible_validation_tasks'])}),flush=True)


if __name__=='__main__':main()
