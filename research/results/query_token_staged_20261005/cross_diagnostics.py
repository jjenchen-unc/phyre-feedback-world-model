"""Offline cross attribution, paired first updates and oracle geometry checks."""
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np

R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'code'))
from attribution_a_metrics import log_score, auc
read=lambda p:json.loads(p.read_text())
def mean(xs):
    xs=[x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None
def first_rank(scores,y):
    hits=np.flatnonzero(y[np.argsort(-scores,kind='stable')])
    return int(hits[0])+1 if len(hits) else None

out={}
for split,dirname in [('cross','evaluation'),('within','within_evaluation')]:
    root=R/dirname;oracle=[];fixed=[];updates=[]
    for p in sorted((root/'q0_s0'/'fixed').glob('*/truth.npz')):
        d=read(p.parent/'summary.json')
        with np.load(p) as z:
            y=z['labels'].astype(bool); sc=log_score(z['distance'],d['rule'])
            assert len(y)==1024
            with np.load(root/'q0_s0'/'closed'/p.parent.name/'scores.npz') as closed:
                np.testing.assert_array_equal(z['indices'],closed['indices'])
                np.testing.assert_array_equal(y,closed['labels'])
        oracle.append(dict(task=d['task'],template=d['task'][:5],positive=int(y.sum()),rank=first_rank(sc,y),auc=auc(y,sc)))
    assert len(oracle)==100
    for seed in range(3):
        for p in (root/f'residual_s{seed}'/'fixed').glob('*/summary.json'):
            d=read(p)
            if not d['context']:continue
            for head in ['original','predicted_mix_s0']:
                b=d['conditions'][head+'__base'];m=d['conditions'][head+'__memory']
                qd=read(root/f'q0_s{seed}'/'fixed'/p.parent.name/'summary.json')
                original=qd['conditions'][head+'__base']
                row=dict(seed=seed,task=d['task'],template=d['task'][:5],head=head,positive=b['n_success'])
                for name,v in [('original',original),('q0',b),('updated',m)]:
                    row[name]=dict(auc=v['auc'],l1=v['all']['latent_l1'],geometry=v['all']['surface_mae'],success_geometry=v['success']['surface_mae'])
                fixed.append(row)
        for p in (root/f'residual_s{seed}'/'closed').glob('*/snapshots.npz'):
            with np.load(p) as z:
                y=z['labels'].astype(bool);ss=z['scores'];q0=z['base_scores']
            if len(ss)<2 or not y.any():continue
            chosen0=int(np.argmax(q0));assert not y[chosen0]
            keep=np.ones(len(y),bool);keep[chosen0]=False
            with np.load(root/f'zero_s{seed}'/'closed'/p.parent.name/'snapshots.npz') as z:
                zs=z['scores'];np.testing.assert_array_equal(zs[0],q0)
                assert len(zs)>1
            row=dict(seed=seed,task=p.parent.name.replace('_',':'),template=p.parent.name[:5])
            for name,sc in [('q0',q0),('residual',ss[1]),('zero',zs[1])]:
                row[name]=dict(auc=auc(y[keep],sc[keep]),rank=first_rank(sc[keep],y[keep]),
                              best_success_margin=float(np.max(sc[y])-np.max(sc[(~y)&keep])))
            updates.append(row)
    result={'oracle_geometry':{},'fixed':{},'matched_first_update':{}}
    for name,subset in [('all',oracle)]+[(t,[r for r in oracle if r['template']==t]) for t in sorted({r['template'] for r in oracle})]:
        result['oracle_geometry'][name]=dict(n=len(subset),has_positive=sum(r['positive']>0 for r in subset),
            s1=sum(r['rank']==1 for r in subset),s9=sum(r['rank'] is not None and r['rank']<=9 for r in subset),mean_auc=mean([r['auc'] for r in subset]))
    for head in ['original','predicted_mix_s0']:
        hh=[r for r in fixed if r['head']==head]
        result['fixed'][head]={}
        for name,rr in [('all',hh)]+[(t,[r for r in hh if r['template']==t]) for t in sorted({r['template'] for r in hh})]:
            result['fixed'][head][name]={'n':len(rr),'auc_n':sum(r['q0']['auc'] is not None for r in rr)}
            for stage in ['original','q0','updated']:
                result['fixed'][head][name][stage]={k:mean([r[stage][k] for r in rr]) for k in ['auc','l1','geometry','success_geometry']}
    for name,rr in [('all',updates)]+[(t,[r for r in updates if r['template']==t]) for t in sorted({r['template'] for r in updates})]:
        result['matched_first_update'][name]={'n':len(rr)}
        for stage in ['q0','residual','zero']:
            result['matched_first_update'][name][stage]={k:mean([r[stage][k] for r in rr]) for k in ['auc','rank','best_success_margin']}
        for metric in ['auc','rank','best_success_margin']:
            result['matched_first_update'][name][metric+'_direction']={
                'increased':sum(r['residual'][metric]>r['q0'][metric] for r in rr),
                'decreased':sum(r['residual'][metric]<r['q0'][metric] for r in rr),
                'equal':sum(r['residual'][metric]==r['q0'][metric] for r in rr)}
    result['oracle_rows']=oracle;result['first_update_rows']=updates
    by_scene=defaultdict(list)
    for r in fixed:
        if r['head']=='predicted_mix_s0' and r['q0']['auc'] is not None:
            by_scene[r['task']].append(r['updated']['auc']-r['q0']['auc'])
    assert all(len(v)==3 for v in by_scene.values())
    x=np.array([np.mean(v) for v in by_scene.values()]);rng=np.random.default_rng(20261007)
    ci=np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975])
    result['fixed_auc_bootstrap']={'n_scenes':len(x),'mean_delta':float(x.mean()),'ci95':ci.tolist(),
        'scope':'Paired scene bootstrap, three seeds kept together; conditional on existing templates and trained models, not independent test evidence'}
    out[split]=result
(R/'CROSS_MECHANISM_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({s:{k:v for k,v in d.items() if not k.endswith('_rows')} for s,d in out.items()},indent=2))
