import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];P=R.parents[1];SOURCE=P/'results/q0_first_feedback_20261018'
read=lambda p:json.loads(Path(p).read_text())
plan=read(R/'PLAN.json');ph=hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest();out={'complete':True,'plan_sha256':ph,'scope':'Frozen-checkpoint closed-loop9attempt interventions; one seed and onefold per protocol','panels':{}}
assert read(R/'ENCODER_VALIDATION.json')['passed']
for panel,tasks in plan['tasks'].items():
    rows={};methods={}
    for method in ['q0','residual','zero','tta']:
        rows[method]={t:read(SOURCE/'evaluation'/panel/method/t.replace(':','_')/'result.json') for t in tasks}
    validation=read(R/'evaluation'/panel/'real'/plan['validation_tasks'][panel].replace(':','_')/'result.json')
    assert validation['complete'] and validation['first_scores_exact'] and validation['model_unchanged']
    for method in [v for v in plan['variants'] if v!='real']:
        rows[method]={}
        for t in tasks:
            folder=R/'evaluation'/panel/method/t.replace(':','_');row=read(folder/'result.json');assert row['complete'] and row['provenance']['plan_sha256']==ph and row['first_scores_exact'] and row['model_unchanged']
            with np.load(folder/'scores.npz') as z,np.load(SOURCE/'evaluation'/panel/'residual'/t.replace(':','_')/'scores.npz') as original:
                np.testing.assert_array_equal(z['indices'],original['indices']);np.testing.assert_array_equal(z['labels'],original['labels']);np.testing.assert_array_equal(z['scores'][0],original['scores'][0]);selected=z['selected'];assert len(set(selected))==len(selected)
                for turn,j in enumerate(selected):
                    scores=z['scores'][turn].copy();scores[selected[:turn]]=-np.inf;assert int(scores.argmax())==j
                    assert row['events'][turn]['local']==int(j) and row['events'][turn]['action_idx']==int(z['indices'][j]) and row['events'][turn]['solved']==bool(z['labels'][j])
                hits=np.flatnonzero(z['labels'][selected]);actual=int(hits[0]+1) if len(hits) else None
                assert actual==row['first_success'];assert len(selected)==(actual or 9)
            rows[method][t]=row
    for method,rs in rows.items():
        first={t:r['first_success'] for t,r in rs.items()};methods[method]={'n':len(first),'counts':{f's{k}':sum(v is not None and v<=k for v in first.values()) for k in [1,3,5,9]},'first_success':first}
        paired={}
        for ref in ['q0','residual']:
            gains=[t for t in tasks if first[t] is not None and rows[ref][t]['first_success'] is None];losses=[t for t in tasks if first[t] is None and rows[ref][t]['first_success'] is not None]
            paired[ref]={'gains':len(gains),'losses':len(losses),'gain_tasks':gains,'loss_tasks':losses}
        methods[method]['paired_s9']=paired
        templates={}
        for template in sorted({t.split(':')[0] for t in tasks}):
            subset=[t for t in tasks if t.startswith(template+':')];templates[template]={'n':len(subset),'solved':sum(first[t] is not None for t in subset)}
        methods[method]['templates']=templates
        if method in plan['variants']:
            methods[method]['seconds']=sum(x['seconds'] for x in rs.values());methods[method]['observation_s']=sum(e['observation_s'] for x in rs.values() for e in x['events']);methods[method]['encoder_s']=sum(e.get('observation_encode_s',0) for x in rs.values() for e in x['events'])
    out['panels'][panel]={'methods':methods}
(R/'SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n');(R/'STATUS.json').write_text(json.dumps({'stage':'complete','complete':True,'plan_sha256':ph},indent=2)+'\n')
print(json.dumps({p:{m:{'counts':x['counts'],'paired_s9':{r:{k:y[k] for k in ['gains','losses']} for r,y in x['paired_s9'].items()}} for m,x in v['methods'].items()} for p,v in out['panels'].items()},indent=2))
