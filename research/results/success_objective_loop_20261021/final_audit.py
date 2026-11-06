"""Independent CPU audits of final adaptive experiments; no runtime imports."""
from pathlib import Path
import json
import hashlib
import numpy as np
from analysis_audit import metric, filescores

R = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def aggregate(rows):
    names = set.intersection(*(set(r) for r in rows.values()))
    return {n: {**{k: sum(r[n][k] for r in rows.values()) for k in ['top1','top3','top5','top8']},
                'auc': float(np.mean([r[n]['auc'] for r in rows.values() if r[n]['auc'] is not None]))}
            for n in sorted(names)}
def paired(rows, a, b):
    return {k: {'gain':sum(r[a][k]>r[b][k] for r in rows.values()),
                'loss':sum(r[a][k]<r[b][k] for r in rows.values())}
            for k in ['top1','top8']}

def amplitude():
    root=R/'update_amplitude';plan=read(root/'PLAN.json');rows={}
    for task in plan['tasks']:
        d=root/'evaluation'/task.replace(':','_');receipt=read(d/'result.json')
        assert receipt['complete'] and receipt['plan']==sha(root/'PLAN.json')
        if not receipt['first_failed']:continue
        assert receipt['reader_unchanged'] and max(receipt['anchor_errors'].values())==0
        z=filescores(d/'scores.npz');ref=filescores(R/'cross_transfer/evaluation'/task.replace(':','_')/'scores.npz')
        for k in ['indices','labels','first']:np.testing.assert_array_equal(z[k],ref[k])
        arrays={k:v for k,v in z.items() if k not in ['indices','labels','first']}
        arrays['q0']=ref['q0']
        rows[task]={k:metric(v,z['labels'],int(z['first'])) for k,v in arrays.items()}
    return dict(complete=True,n=len(plan['tasks']),first_failed=len(rows),summary=aggregate(rows),
                paired={k:paired(rows,k,'q0') for k in next(iter(rows.values())) if k!='q0'})

def readouts():
    models=read(R/'readout/MODELS.json');tasks=read(R/'cross_transfer/EXTENSION_PLAN.json')['all_tasks'];errors={k:0. for k in models}
    for t in tasks:
        z=filescores(R/'cross_transfer/data'/t.replace(':','_')/'q0.npz')
        ref=filescores(R/'cross_transfer/evaluation'/t.replace(':','_')/'scores.npz')
        for k in ['indices','labels','first']:np.testing.assert_array_equal(z[k],ref[k])
        d=z['distance'];soft=1/(1+np.exp(np.clip(d.astype(float)-4,-700,700)))
        longest=[]
        for row in d<4:
            best=run=0
            for v in row:
                run=run+1 if v else 0;best=max(best,run)
            longest.append(best/32)
        x=np.column_stack([soft,z['scores'],d.min(1)/256,d.mean(1)/256,d.max(1)/256,longest])
        for name,m in models.items():
            score=((x-m['mean'])/m['std'])@m['weight']+m['bias']
            err=float(abs(score-ref[name]).max());errors[name]=max(errors[name],err)
            assert err<1e-10,(t,name,err)
            assert metric(score,z['labels'],int(z['first']))==metric(ref[name],z['labels'],int(z['first']))
    return dict(complete=True,tasks=len(tasks),max_score_error=errors,features=37)

def contact():
    root=R/'contact_training';p=read(root/'PLAN.json');parent=read(R/'PLAN.json');out={}
    assert p['source_sha']==sha(root/'experiment.py') and p['parent']==sha(R/'PLAN.json')
    control=read(R/'models/seed7919/prediction_residual/complete.json')
    for arm in p['arms']:
        ck=read(root/'models'/arm/'complete.json');assert ck['complete'] and ck['steps']==128 and ck['plan']==sha(root/'PLAN.json')
        for k in ['initial_sha','reader_sha','query_schedule_sha']:assert ck[k]==control[k],(arm,k)
    for stage in ['discovery','confirmation','cross100']:
        tasks=read(R/'cross_transfer/EXTENSION_PLAN.json')['all_tasks'] if stage=='cross100' else parent[stage]
        rows={};losses=[]
        for t in tasks:
            name=t.replace(':','_');d=root/'evaluation'/stage/name;receipt=read(d/'result.json')
            assert receipt['complete'] and receipt['plan']==sha(root/'PLAN.json')
            path=R/'cross_transfer/evaluation'/name/'scores.npz' if stage=='cross100' else R/'evaluation/seed7919'/stage/name/'scores.npz'
            ref=filescores(path);first=int(ref['first']);failed=not ref['labels'][first]
            assert failed==receipt['first_failed'] and first==int(np.argmax(ref['q0']))
            if not failed:continue
            assert receipt['reader_unchanged'];z=filescores(d/'scores.npz')
            for k in ['indices','labels','first']:np.testing.assert_array_equal(z[k],ref[k])
            arrays={k:z[k] for k in p['arms']};arrays['q0']=ref['q0']
            for cond in ['residual','zero']:
                key='prediction_'+cond;arrays[key]=ref[key+'_s7919' if stage=='cross100' else key]
            rows[t]={k:metric(v,z['labels'],first) for k,v in arrays.items()};losses.append(receipt['losses'])
        pairs={}
        for arm in p['arms']:
            for ctrl in ['q0','prediction_'+arm.split('_')[1],arm.split('_')[0]+'_zero']:
                if ctrl!=arm:pairs[arm+' vs '+ctrl]=paired(rows,arm,ctrl)
        loss_summary={}
        if stage!='cross100':
            for a in p['arms']:loss_summary[a]={k:float(np.mean([v[a][k] for v in losses])) for k in losses[0][a]}
        out[stage]=dict(n=len(tasks),first_failed=len(rows),summary=aggregate(rows),paired=pairs,losses=loss_summary,per_scene=rows)
    return dict(complete=True,panels=out)

def loss_controls():
    root=R/'contact_training';raw=read(root/'LOSS_CONTROLS.json')
    assert raw['complete'] and raw['reader_unchanged']
    assert raw['source_sha']==sha(root/'loss_controls.py') and raw['plan']==sha(root/'PLAN.json')
    result={}
    for stage,rows in raw['records'].items():
        assert len(rows)==(12 if stage=='discovery' else 10)
        names=list(next(iter(rows.values())))
        for task,values in rows.items():
            check=read(root/'evaluation'/stage/task.replace(':','_')/'result.json')['losses']
            for arm,loss in check.items():
                for k,v in loss.items():assert abs(v-values[arm][k])<1e-7,(task,arm,k,v,values[arm][k])
            old=read(R/'evaluation/seed7919'/stage/task.replace(':','_')/'result.json')
            # Original ranking/loss receipts are retained; loss computation here is
            # a separate forward replay, not an independent geometry implementation.
        result[stage]={a:{k:float(np.mean([v[a][k] for v in rows.values()])) for k in ['l1','geometry','contact']} for a in names}
    return dict(complete=True,summary=result,scope='Matched frozen-checkpoint forward replay. Original geometry code reused; not independent geometry derivation.')

if __name__=='__main__':
    report=dict(amplitude=amplitude(),readouts=readouts())
    if all((R/'contact_training/models'/a/'complete.json').exists() for a in ['contact01_residual','contact01_zero','contact1_residual','contact1_zero']):
        report['contact']=contact()
    if (R/'contact_training/LOSS_CONTROLS.json').exists():report['loss_controls']=loss_controls()
    (R/'FINAL_INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,result in report.items():
        print(name, {k:v for k,v in result.items() if k not in ['paired','panels']})
        for stage,p in result.get('panels',{}).items():print(stage,p['summary'])
