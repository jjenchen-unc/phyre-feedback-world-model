"""Independent score audit; no training or inference module imports."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
R=Path(__file__).resolve().parent;OLD=R.parent/'success_objective_loop_20261021'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def metrics(s,y,j):
    ids=[i for i in range(len(y)) if i!=j];order=sorted(ids,key=lambda i:(-float(s[i]),i));pos=s[ids][y[ids]];neg=s[ids][~y[ids]]
    return {**{f'top{k}':int(any(y[i] for i in order[:k])) for k in [1,3,5,8]},'auc':float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean()) if len(pos) and len(neg) else None}
def aggregate(rows,total):
    names=sorted(set.intersection(*(set(v) for v in rows.values())));out={}
    for n in names:
        counts={k:sum(v[n][k] for v in rows.values()) for k in ['top1','top3','top5','top8']};auc=[v[n]['auc'] for v in rows.values() if v[n]['auc'] is not None]
        out[n]={**counts,'auc':float(np.mean(auc)) if auc else None,'S2_count':total-len(rows)+counts['top1']}
    return out
def paired(rows,a,b):
    return {k:dict(gain=sum(v[a][k]>v[b][k] for v in rows.values()),loss=sum(v[a][k]<v[b][k] for v in rows.values())) for k in ['top1','top3','top5','top8']}
def audit():
    p=read(R/'PLAN.json');meta=read(R/'DATA.json');assert meta['plan']==sha(R/'PLAN.json') and sha(R/'study.py')==p['source_sha'];assert read(R/'SMOKE.json')['passed']
    allts=[t for ts in p['panels'].values() for t in ts];assert len(allts)==len(set(allts))
    counts={};init={};manifests={}
    for seed in p['seeds']:
        receipts=[read(R/'runs'/f'{seed}_{a}'/'complete.json') for a in p['arms']]
        assert all(v['complete'] and v['plan']==sha(R/'PLAN.json') and v['data']==sha(R/'DATA.json') and v['steps']==512 for v in receipts)
        assert len({v['initial_sha'] for v in receipts})==len({v['query_labels_sha'] for v in receipts})==1
        init[str(seed)]=receipts[0]['initial_sha']
    assert len(set(init.values()))==2
    panels={}
    for stage in ['discovery','confirmation','cross100']:
        rows={}
        for task in p['panels'][stage]:
            if not meta['records'][task]['failed']:continue
            name=task.replace(':','_');raw=load(R/'data'/name/'features.npz');y=raw['labels'];j=int(raw['first']);assert j==int(raw['q0'].argmax()) and not y[j]
            arrays={'q0':raw['q0'],'static':raw['baseline']}
            prior=load(OLD/('cross_transfer/evaluation' if stage=='cross100' else 'readout/evaluation/'+stage)/name/'scores.npz')
            for k in ['labels','indices','first']:np.testing.assert_array_equal(raw[k],prior[k])
            np.testing.assert_allclose(raw['baseline'],prior['query8_readout'],rtol=0,atol=1e-10)
            for seed in p['seeds']:
                for arm in p['arms']:
                    f=R/'runs'/f'{seed}_{arm}'/'evaluation'/stage/name/'scores.npz';z=load(f);manifests[str(f.relative_to(R))]=sha(f)
                    for k in ['labels','indices','first']:np.testing.assert_array_equal(z[k],raw[k])
                    np.testing.assert_array_equal(z['q0'],raw['q0']);np.testing.assert_array_equal(z['static'],raw['baseline'])
                    arrays[f'{seed}_{arm}']=z['actual'];assert np.isfinite(z['actual']).all()
                    if arm=='real':
                        for i in range(5):arrays[f'{seed}_shuffle{i}']=z[f'shuffle{i}']
            rows[task]={n:metrics(s,y,j) for n,s in arrays.items()}
        summary=aggregate(rows,len(p['panels'][stage]));pair={}
        for seed in p['seeds']:
            for b in ['q0','static',f'{seed}_zero',f'{seed}_query_only']:
                pair[f'{seed}_real vs {b}']=paired(rows,f'{seed}_real',b)
        panels[stage]=dict(n=len(p['panels'][stage]),first_failed=len(rows),summary=summary,paired=pair,per_scene=rows)
    cross=panels['cross100']['summary'];gates={}
    for seed in p['seeds']:
        real=cross[f'{seed}_real'];shuffle_mean=float(np.mean([cross[f'{seed}_shuffle{i}']['top1'] for i in range(5)]))
        gates[str(seed)]=dict(top1_beats_static=real['top1']>cross['static']['top1'],top1_beats_zero=real['top1']>cross[f'{seed}_zero']['top1'],top1_beats_query_only=real['top1']>cross[f'{seed}_query_only']['top1'],no_top5_regression=real['top5']>=cross['static']['top5'],top1_beats_mean_shuffle=real['top1']>shuffle_mean,shuffle_mean_top1=shuffle_mean)
    passed=all(all(v for k,v in g.items() if k!='shuffle_mean_top1') for g in gates.values())
    retained={stage:dict(median=float(np.median([meta['retained_by_scene'][t] for t in ts if t in meta['retained_by_scene']])),minimum=float(min(meta['retained_by_scene'][t] for t in ts if t in meta['retained_by_scene']))) for stage,ts in p['panels'].items()}
    report=dict(complete=True,initial_sha=init,panels=panels,gate=gates,expansion_gate_passed=passed,fit_projection_energy=meta['fit_projection_energy'],projection_energy_by_stage=retained,raw_scores_hashes=manifests,
        scope='Independent cachedscore/label/rank verification. SamefirstQ0failure histories; conditionaltopk afteronefeedback and actualS2 reconstructed fromcachedlabels. No inference about repeatedfeedbackS9.')
    write=R/'AUDIT.json';write.write_text(json.dumps(report,indent=2)+'\n')
    with (R/'CROSS100_TABLE.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['method','top1','top3','top5','top8','auc','S2_count']);writer.writeheader();writer.writerows(dict(method=k,**v) for k,v in cross.items())
    print('GATE',passed,gates,'projection',retained)
    for stage,panel in panels.items():print(stage,panel['summary'])
if __name__=='__main__':audit()
