"""Independent metrics, feature construction and neural replay for ablations."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
from scipy.special import ndtr
R=Path(__file__).resolve().parent;PRE=R.parent/'feedback_readout_loop_20261021'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def metric(s,y,j):
    ids=[i for i in range(len(y)) if i!=j];order=sorted(ids,key=lambda i:(-float(s[i]),i));pos=s[ids][y[ids]];neg=s[ids][~y[ids]]
    return {**{f'top{k}':int(any(y[i] for i in order[:k])) for k in [1,3,5,8]},'auc':float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean()) if len(pos) and len(neg) else None}
def summarize(rows,total):
    out={}
    for n in sorted(next(iter(rows.values()))):
        v=[r[n] for r in rows.values()];a=[m['auc'] for m in v if m['auc'] is not None];counts={k:sum(m[k] for m in v) for k in ['top1','top3','top5','top8']}
        out[n]={**counts,'auc':float(np.mean(a)) if a else None,'S2_count':total-len(rows)+counts['top1']}
    return out
def paired(rows,a,b):
    return {k:dict(gain=sum(r[a][k]>r[b][k] for r in rows.values()),loss=sum(r[a][k]<r[b][k] for r in rows.values())) for k in ['top1','top5','top8']}
def features(z,arm,norm,donor=None):
    # Separate featureassembly from trained implementation.
    s=z if donor is None else donor;k=int(s['first']);n=len(z['q']);x=np.zeros((n,131));x[:,:40]=np.column_stack([z['q'],z['actions']])
    if arm in ['geometry_abs','geometry_both','absolute_both']:x[:,40:77]=s['q'][k]
    if arm in ['action_abs','action_both','absolute_both','query_distance_action']:x[:,77:80]=s['actions'][k]
    if arm in ['geometry_rel','geometry_both','relative_both']:x[:,80:112]=(z['distance']-s['distance'][k])/256
    if arm.startswith('query_distance'):x[:,80:112]=z['distance']/256
    if arm in ['action_rel','action_both','relative_both','query_distance_action']:x[:,112:115]=z['actions']-s['actions'][k]
    active=np.zeros(131,dtype=bool);active[:40]=True
    ranges={'action_abs':[(77,80)],'action_rel':[(112,115)],'action_both':[(77,80),(112,115)],'geometry_abs':[(40,77)],'geometry_rel':[(80,112)],'geometry_both':[(40,77),(80,112)],'absolute_both':[(40,80)],'relative_both':[(80,115)],'query_distance':[(80,112)],'query_distance_action':[(77,115)]}
    for a,b in ranges[arm]:active[a:b]=True
    x[:,active]=np.clip((x[:,active]-np.asarray(norm['mean'])[active])/np.asarray(norm['std'])[active],-10,10);x[:,~active]=0
    return x.astype(np.float32)
def forward(x,w):
    for l in ['0','2']:x=x@w[l+'.weight'].T+w[l+'.bias'];x=x*ndtr(x)
    return (x@w['4.weight'].T+w['4.bias']).ravel()
def audit():
    p=read(R/'PLAN.json');assert p['source_sha']==sha(R/'study.py');assert read(R/'SMOKE.json')['passed'] and read(R/'CONTROL_REPLAY.json')['passed'];meta=read(PRE/'DATA.json');norms=read(R/'NORMALIZATIONS.json');cache={}
    for t,r in meta['records'].items():
        if r['failed']:
            f=PRE/'data'/t.replace(':','_')/'features.npz';assert r['data_sha']==sha(f);cache[t]=load(f)
    xs=np.concatenate([cache[t]['distance'][cache[t]['query']]/256 for t in p['panels']['fit'] if t in cache]);oldnorm=read(PRE/'NORMALIZATION.json')
    for arm,n in norms.items():
        if arm.startswith('query_distance'):
            np.testing.assert_array_equal(n['mean'][80:112],xs.mean(0));np.testing.assert_array_equal(n['std'][80:112],np.maximum(xs.std(0),1e-5))
            for k in ['mean','std']:np.testing.assert_array_equal(n[k][:80]+n[k][112:],oldnorm[k][:80]+oldnorm[k][112:])
        else:
            for k in ['mean','std']:np.testing.assert_array_equal(n[k],oldnorm[k])
    for seed in p['seeds']:
        ref=read(PRE/'runs'/f'{seed}_zero'/'complete.json')
        for arm in p['arms']:
            ck=read(R/'runs'/f'{seed}_{arm}'/'complete.json');assert ck['complete'] and ck['plan']==sha(R/'PLAN.json') and ck['steps']==512
            for k in ['initial_sha','query_labels_sha']:assert ck[k]==ref[k],(seed,arm,k)
    panels={};replay=[];vectors=0
    for stage in ['discovery','confirmation','cross100']:
        rows={};selected=set();seen=set()
        for t in sorted(p['panels'][stage]):
            if t in cache and t[:5] not in seen:selected.add(t);seen.add(t[:5])
        for task in p['panels'][stage]:
            if task not in cache:continue
            z=cache[task];name=task.replace(':','_');y=z['labels'];j=int(z['first']);assert not y[j] and j==int(np.argmax(z['q0']))
            arrays={'q0':z['q0'],'static':z['baseline']}
            for seed in p['seeds']:
              for arm in ['zero','query_only']:
                q=load(PRE/'runs'/f'{seed}_{arm}'/'evaluation'/stage/name/'scores.npz')
                for k in ['labels','indices','first']:np.testing.assert_array_equal(q[k],z[k])
                arrays[f'{seed}_{arm}']=q['actual']
              for arm in p['arms']:
                folder=R/'runs'/f'{seed}_{arm}';q=load(folder/'evaluation'/stage/name/'scores.npz')
                for k in ['labels','indices','first']:np.testing.assert_array_equal(q[k],z[k])
                np.testing.assert_array_equal(q['static'],z['baseline']);np.testing.assert_array_equal(q['q0'],z['q0']);assert np.isfinite(q['actual']).all()
                arrays[f'{seed}_{arm}']=q['actual']
                if stage=='cross100' and arm!='query_distance':
                    for i in range(5):arrays[f'{seed}_{arm}_shuffle{i}']=q[f'shuffle{i}']
                if task in selected:
                    w=load(folder/'weights.npz')
                    for mode in ['actual']+(['shuffle0'] if stage=='cross100' and arm!='query_distance' else []):
                        donor=cache[meta['shuffle'][stage][0][task]] if mode=='shuffle0' else None;v=z['baseline']+forward(features(z,arm,norms[arm],donor),w);err=float(abs(v-q[mode]).max());assert err<1e-4,(seed,arm,task,err)
                        actual=metric(v,y,j);expected=metric(q[mode],y,j)
                        for k in ['top1','top3','top5','top8']:assert actual[k]==expected[k],(task,arm,k)
                        replay.append(err)
            rows[task]={k:metric(v,y,j) for k,v in arrays.items()};vectors+=len(arrays)
        summary=summarize(rows,len(p['panels'][stage]));pairs={}
        for seed in p['seeds']:
            for arm in p['arms']+['zero']:
                for ctrl in ['query_only','query_distance','zero']:
                    if arm!=ctrl:pairs[f'{seed}_{arm} vs {ctrl}']=paired(rows,f'{seed}_{arm}',f'{seed}_{ctrl}')
        panels[stage]=dict(n=len(p['panels'][stage]),first_failed=len(rows),summary=summary,paired=pairs,per_scene=rows)
    hashes=read(R/'REMOTE_HASHES.json');assert len(hashes['checkpoints'])==20
    for path,h in hashes['files'].items():assert sha(R/path)==h,path
    report=dict(complete=True,panels=panels,raw_vectors_recomputed=vectors,independent_replay_vectors=len(replay),maximum_replay_error=max(replay),topk_replay_agreement=True,train_only_normalization_exact=True,local_remote_files_matched=len(hashes['files']),initialization_and_labels_matched=True,
        interpretation='Allconditionsreported. Samefirstfailurehistory;nextkstaticafteroneupdate, S2 reconstructed. Featureablation isnotproof ofcausalusefulphysicsfeedback; allcontextdescriptorsarepredicted/pre-attempt. IndependentGELU/feature/metricreplay; noimportoftrainingcode.')
    (R/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    with (R/'CROSS100_TABLE.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=['method','top1','top3','top5','top8','auc','S2_count']);wr.writeheader();wr.writerows(dict(method=k,**v) for k,v in panels['cross100']['summary'].items() if 'shuffle' not in k)
    print('COMPLETE',vectors,'REPLAY',len(replay),'MAXERR',max(replay))
    for stage,panel in panels.items():print(stage,{k:v for k,v in panel['summary'].items() if 'shuffle' not in k})
if __name__=='__main__':audit()
