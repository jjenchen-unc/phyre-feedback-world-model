"""Paired fixed-history measurements; not a new closed-loop success estimate."""
import hashlib,json
from pathlib import Path
from statistics import mean,median
import numpy as np
R=Path(__file__).resolve().parents[1]
plan=json.loads((R/'PLAN.json').read_text());ph=hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest()

def differences(rows,v,ref):
    pairs=[(r['variants'][v]['ranking'],r['variants'][ref]['ranking']) for r in rows]
    auc=[a['pair_auc']-b['pair_auc'] for a,b in pairs if a['pair_auc'] is not None and b['pair_auc'] is not None]
    ranks=[b['best_success_rank']-a['best_success_rank'] for a,b in pairs if a['best_success_rank'] is not None and b['best_success_rank'] is not None]
    return {'n':len(pairs),'next_action_changed':sum(a['next_local']!=b['next_local'] for a,b in pairs),
        'next_success_gain':sum(a['next_success'] and not b['next_success'] for a,b in pairs),
        'next_success_loss':sum(b['next_success'] and not a['next_success'] for a,b in pairs),
        'auc_defined':len(auc),'auc_mean_change':mean(auc) if auc else None,'auc_median_change':median(auc) if auc else None,
        'auc_improve':sum(x>0 for x in auc),'auc_worsen':sum(x<0 for x in auc),
        'best_success_rank_improve':sum(x>0 for x in ranks),'best_success_rank_worsen':sum(x<0 for x in ranks),
        'best_success_rank_defined':len(ranks),'best_success_rank_median_improvement':median(ranks) if ranks else None}

out={'complete':True,'plan_sha256':ph,'scope':'Fixed history, one-step intervention; first/later separated. Correlated states from same scenes. No S@9.',
     'limits':plan['limits'],'panels':{}}
verified=0
for panel,tasks in plan['tasks'].items():
    rows=[]
    for task in tasks:
        folder=R/'cases'/panel/task.replace(':','_');row=json.loads((folder/'result.json').read_text())
        assert row['complete'] and row['plan_sha256']==ph
        for c in row['cases']:
            assert c['instrumentation_exact'] and c['original_before_after_scores_exact'] and c['parent_real_zero_exact']
            with np.load(folder/f"update_{c['update_number']}_scores.npz") as z:
                ids=np.flatnonzero(~z['tried']);label=z['labels'];ref=ids[np.argsort(-z['real'][ids],kind='stable')]
                np.testing.assert_array_equal(z['step_0'],z['before'])
                for v in plan['variants']:
                    scores=z[v];m=c['variants'][v]['ranking'];order=ids[np.argsort(-scores[ids],kind='stable')]
                    assert m['next_local']==int(order[0]) and m['next_success']==bool(label[order[0]])
                    assert m['top1_changed']==bool(order[0]!=ref[0])
                    pos=scores[ids][label[ids]];neg=np.sort(scores[ids][~label[ids]])
                    if len(pos) and len(neg):
                        auc=float(np.mean((np.searchsorted(neg,pos,side='left')+np.searchsorted(neg,pos,side='right'))/(2*len(neg))))
                        assert abs(auc-m['pair_auc'])<1e-12
                    else:assert m['pair_auc'] is None
                    hits=np.flatnonzero(label[order]);assert m['best_success_rank']==(int(hits[0]+1) if len(hits) else None)
                    verified+=1
            rows.append(c)
    groups={}
    for name,rs in [('first',[x for x in rows if x['update_number']==1]),('later',[x for x in rows if x['update_number']>1])]:
        g={'n':len(rs),'pooled_mean_energy_fraction_median':median(x['variants']['real']['components']['pooled_mean_energy_fraction'] for x in rs),'variants':{}}
        for v in plan['variants']:
            vs=[x['variants'][v] for x in rs]
            g['variants'][v]={'update_relative_difference_vs_real_median':median(x['update_change_vs_real']['relative_l2'] for x in vs),
                'next_successes':sum(x['ranking']['next_success'] for x in vs),'top10_overlap_vs_real_median':median(x['ranking']['top10_overlap'] for x in vs),
                'vs_no_update':differences(rs,v,'step_0'),'vs_real':differences(rs,v,'real'),'vs_zero':differences(rs,v,'zero_residual')}
        groups[name]=g
    out['panels'][panel]={'tasks':len(tasks),'groups':groups}
out['independently_verified_score_arrays']=verified
(R/'SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
(R/'STATUS.json').write_text(json.dumps({'stage':'complete','gpu_diagnostic_complete':True,'plan_sha256':ph,'independently_verified_score_arrays':verified},indent=2)+'\n')
print(json.dumps({'complete':True,'arrays':verified,'panels':{p:{g:{v:x['vs_no_update'] for v,x in z['variants'].items()} for g,z in y['groups'].items()} for p,y in out['panels'].items()}},indent=2))
