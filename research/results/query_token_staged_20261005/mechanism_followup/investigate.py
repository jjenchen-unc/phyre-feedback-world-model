"""Reuse matched raw outputs to separate observed-latent readout from prediction.

All interventions use future labels/geometry OFFLINE; none are deployable policies.
"""
import json, hashlib, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

HERE=Path(__file__).resolve().parent
R=HERE.parent
OLD=R.parent/'probe_adaptation_cross_20261001'
sys.path.insert(0,str(R/'code'))
from attribution_a_metrics import log_score, auc
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p) as d:return {k:d[k] for k in ['indices','labels','query_mask','distance','log_score'] if k in d}
def avg(x):
    x=[a for a in x if a is not None]
    return float(np.mean(x)) if x else None
def rank(y,s):
    h=np.flatnonzero(y[np.argsort(-s,kind='stable')])
    return int(h[0])+1 if len(h) else None
def metric(y,s):
    r=rank(y,s)
    return dict(rank=r,s1=r==1,s9=r is not None and r<=9,auc=auc(y,s))
def alignment(d,b,t,mask):
    if not mask.any():return None
    delta=(d[mask,1:]-b[mask,1:]).astype(float).ravel()
    needed=(t[mask,1:]-b[mask,1:]).astype(float).ravel()
    norm=np.linalg.norm(delta)*np.linalg.norm(needed)
    return float(delta@needed/norm) if norm else None

m=read(R/'manifest.json');om=read(OLD/'manifest.json');am=read(R.parent/'attribution_stage_a_cross_20260930'/'manifest.json')
assert m['cross_pack']==am['pack'] and m['cross_meta_sha256']==am['meta_sha256']
assert m['probe_sha256']==om['probe_sha256']==am['probe_sha256']
rows=[];oracles=[];audited=0
for folder in sorted((R/'evaluation'/'q0_s0'/'fixed').iterdir()):
    if not (folder/'summary.json').exists():continue
    name=folder.name;old=OLD/'evaluation'/'cross'/name
    spec=read(folder/'summary.json');ospec=read(old/'summary.json')
    assert ospec['provenance']['manifest']==sha(OLD/'manifest.json')
    assert spec['rule']==ospec['rule'] and spec['task']==ospec['task']
    truth=load(folder/'truth.npz');ot=load(old/'truth.npz')
    for k in ['indices','labels','distance']:np.testing.assert_array_equal(truth[k],ot[k])
    y=truth['labels'].astype(bool);td=truth['distance'];rule=spec['rule'];mask=truth['query_mask'].astype(bool)
    oracle=metric(y,log_score(td,rule));oracle.update(task=spec['task'],template=name[:5],head='true_geometry',n_positive=int(y.sum()))
    oracles.append(oracle)
    for head in ['original','predicted_mix_s0']:
        if head!='original':assert ospec['provenance']['heads'][head]==m['heads'][head]
        obs=load(old/(head+'__observed.npz'));oldbase=load(old/(head+'__base.npz'));newbase=load(folder/(head+'__base.npz'))
        for k in ['distance','log_score']:np.testing.assert_array_equal(oldbase[k],newbase[k])
        np.testing.assert_array_equal(obs['log_score'],log_score(obs['distance'],rule))
        omet=metric(y,obs['log_score']);omet.update(task=spec['task'],template=name[:5],head=head,n_positive=int(y.sum()),
            all_geometry=float(np.abs(obs['distance'][:,1:]-td[:,1:]).mean()),
            success_geometry=float(np.abs(obs['distance'][y,1:]-td[y,1:]).mean()) if y.any() else None)
        oracles.append(omet)
        for seed in range(3):
            f=R/'evaluation'/f'residual_s{seed}'/'fixed'/name
            ss=read(f/'summary.json');t=load(f/'truth.npz')
            for k in ['indices','labels','query_mask','distance']:np.testing.assert_array_equal(t[k],truth[k])
            b=load(f/(head+'__base.npz'));u=load(f/(head+'__memory.npz'))
            q0=load(R/'evaluation'/f'q0_s{seed}'/'fixed'/name/(head+'__memory.npz'))
            for k in ['distance','log_score']:np.testing.assert_array_equal(b[k],q0[k])
            np.testing.assert_array_equal(b['log_score'],log_score(b['distance'],rule))
            np.testing.assert_array_equal(u['log_score'],log_score(u['distance'],rule))
            audited+=1
            # Full-pool baseline remains comparable to main S@9; fixed support is diagnostic only.
            row=dict(task=spec['task'],template=name[:5],seed=seed,head=head,has_support=bool(ss['context']),
                n_positive=int(y.sum()),original=metric(y,newbase['log_score']),q0_full=metric(y,b['log_score']))
            if not ss['context']:
                np.testing.assert_array_equal(b['log_score'],u['log_score'])
                rows.append(row);continue
            for method,data in [('q0',b),('updated',u),('observed',obs),('truth',{'distance':td,'log_score':log_score(td,rule)})]:
                row[method]=metric(y[mask],data['log_score'][mask])
                row[method]['geometry']=float(np.abs(data['distance'][mask,1:]-td[mask,1:]).mean())
                row[method]['success_geometry']=float(np.abs(data['distance'][mask&y,1:]-td[mask&y,1:]).mean()) if (mask&y).any() else None
            row['alignment_to_truth']={g:alignment(u['distance'],b['distance'],td,mask&subset) for g,subset in [('all',np.ones(len(y),bool)),('success',y),('failure',~y)]}
            ctx=np.isin(truth['indices'],ss['context'])
            assert ctx.sum()==1 and not y[ctx].any()
            row['support_needed_distance_change']=float((td[ctx,1:]-b['distance'][ctx,1:]).mean())
            row['alignment_to_observed']={g:alignment(u['distance'],b['distance'],obs['distance'],mask&subset) for g,subset in [('all',np.ones(len(y),bool)),('success',y),('failure',~y)]}
            row['signed_geometry']={}
            # Use the scorer's existing soft contact weights, no new Near cutoff.
            logw=-np.logaddexp(0.,(td[:,1:].astype(float)-float(rule[1]))/float(rule[2]))
            weights=np.exp(logw-logw.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
            for group,sel in [('success',mask&y),('failure',mask&~y)]:
                if not sel.any():continue
                delta=u['distance'][sel,1:]-b['distance'][sel,1:]
                need=td[sel,1:]-b['distance'][sel,1:]
                row['signed_geometry'][group]=dict(mean_distance_change=float(delta.mean()),mean_needed_change=float(need.mean()),
                    at_truth_score_times_change=float((weights[sel]*delta).sum(1).mean()),
                    at_truth_score_times_needed=float((weights[sel]*need).sum(1).mean()))
            if (mask&y).any() and (mask&~y).any():
                # Lock identities before update to avoid comparing changing maxima.
                pos=int(np.argmax(np.where(mask&y,b['log_score'],-np.inf)))
                neg=int(np.argmax(np.where(mask&~y,b['log_score'],-np.inf)))
                pair={method:dict(success=float(dat['log_score'][pos]),failure=float(dat['log_score'][neg]),
                    margin=float(dat['log_score'][pos]-dat['log_score'][neg])) for method,dat in [('q0',b),('updated',u),('observed',obs)]}
                ds=pair['updated']['success']-pair['q0']['success'];df=pair['updated']['failure']-pair['q0']['failure']
                pair.update(success_local=pos,failure_local=neg,success_score_change=ds,failure_score_change=df,margin_change=ds-df)
                assert np.isclose(ds-df,pair['updated']['margin']-pair['q0']['margin'])
                row['locked_pair']=pair
                # Oracle replacements isolate which subset of decoded geometry limits ranking.
                row['oracle_repairs']={}
                for ref,refscore in [('truth',log_score(td,rule)),('observed',obs['log_score'])]:
                    for group,sel in [('success',mask&y),('failure',mask&~y)]:
                        score=u['log_score'].copy();score[sel]=refscore[sel]
                        row['oracle_repairs'][ref+'_'+group]=metric(y[mask],score[mask])
            rows.append(row)
assert audited==600 and len(oracles)==300
out=dict(raw_reuse_audit_passed=True,audited_seed_scene_heads=audited,
    provenance={'staged_manifest':sha(R/'manifest.json'),'observed_manifest':sha(OLD/'manifest.json'),
                'cross_meta_sha256':m['cross_meta_sha256'],'head_sha256':m['heads']['predicted_mix_s0'],'script_sha256':sha(Path(__file__))},
    limits=['Reused cross development panel; no new test or trained model','Observed latent bypasses predictor, not encoder or probe',
            'Oracle geometry/label repairs use future truth and cannot deploy','Fixed original-policy failed support differs from actual Q0 policy support',
            'Seeds repeated on same scenes; no independent 300-scene claim'],oracle={},fixed={},rows=rows,oracle_rows=oracles)
for group in ['all','00000','00002','00007','00014']:
    oo=[r for r in oracles if group=='all' or r['template']==group]
    out['oracle'][group]={}
    for head in ['true_geometry','original','predicted_mix_s0']:
        rr=[r for r in oo if r['head']==head]
        out['oracle'][group][head]=dict(n=len(rr),s1=sum(r['s1'] for r in rr),s9=sum(r['s9'] for r in rr),auc=avg([r['auc'] for r in rr]),
            success_geometry=avg([r.get('success_geometry') for r in rr]))
    out['fixed'][group]={}
    for head in ['original','predicted_mix_s0']:
        rr=[r for r in rows if r['head']==head and r['has_support'] and (group=='all' or r['template']==group)]
        pairs=[r['locked_pair'] for r in rr if 'locked_pair' in r]
        out['fixed'][group][head]=dict(n=len(rr),n_pairs=len(pairs))
        dest=out['fixed'][group][head]
        dest['metrics']={method:{k:avg([r[method][k] for r in rr]) for k in ['auc','geometry','success_geometry']} for method in ['q0','updated','observed','truth']}
        dest['pairs']={k:avg([p[k] for p in pairs]) for k in ['success_score_change','failure_score_change','margin_change']}
        dest['pairs']['worse']=sum(p['margin_change']<0 for p in pairs)
        dest['pairs']['worse_with_success_score_drop']=sum(p['margin_change']<0 and p['success_score_change']<0 for p in pairs)
        dest['pairs']['worse_with_failure_score_rise']=sum(p['margin_change']<0 and p['failure_score_change']>0 for p in pairs)
        dest['alignment']={target:{g:avg([r[target][g] for r in rr]) for g in ['all','success','failure']} for target in ['alignment_to_truth','alignment_to_observed']}
        dest['signed_geometry']={g:{k:avg([r['signed_geometry'][g][k] for r in rr if g in r['signed_geometry']]) for k in ['mean_distance_change','mean_needed_change','at_truth_score_times_change','at_truth_score_times_needed']} for g in ['success','failure']}
        positive=[r for r in rr if r['n_positive']]
        dest['alignment_matched_positive_scenes']={target:{g:avg([r[target][g] for r in positive]) for g in ['success','failure']} for target in ['alignment_to_truth','alignment_to_observed']}
        dest['success_geometry_worse']=sum(r['updated']['success_geometry']>r['q0']['success_geometry'] for r in positive)
        dest['success_alignment_negative']=sum(r['alignment_to_truth']['success']<0 for r in positive if r['alignment_to_truth']['success'] is not None)
        dest['support_transfer']={'n_positive_scenes':len(positive),
            'support_needs_farther_success_needs_closer':sum(r['support_needed_distance_change']>0 and r['signed_geometry']['success']['mean_needed_change']<0 for r in positive),
            'above_and_success_updated_farther':sum(r['support_needed_distance_change']>0 and r['signed_geometry']['success']['mean_needed_change']<0 and r['signed_geometry']['success']['mean_distance_change']>0 for r in positive)}
        repair=[r for r in rr if 'oracle_repairs' in r]
        dest['oracle_repairs']={key:sum(r['oracle_repairs'][key]['s9'] for r in repair) for key in ['truth_success','truth_failure','observed_success','observed_failure']}
        dest['oracle_repairs'].update(n=len(repair),updated_s9=sum(r['updated']['s9'] for r in repair),q0_s9=sum(r['q0']['s9'] for r in repair),observed_s9=sum(r['observed']['s9'] for r in repair),truth_s9=sum(r['truth']['s9'] for r in repair))
(HERE/'RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','oracle_rows']},indent=2))
