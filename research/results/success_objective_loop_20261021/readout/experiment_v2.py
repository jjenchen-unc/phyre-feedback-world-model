"""Round2a: isolate frozen Q0 predictor/probe from success-readout fitting."""
import argparse,sys,hashlib,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
from study import *
RR=R/'readout'


def register():
    plan=checked()
    p=dict(parent_plan=sha(R/'PLAN.json'),source_sha=sha(Path(__file__)),scorer_sha=sha(R/'code/scorer_torch.py'),
      trigger='Frozen affine calibration on bounded geometry score has probability ceiling .1178; compare richer readout and data coverage separately, without changing predictor/updater.',
      arms=['scalar_calibration','query8_readout','fullpool_readout'],
      features='32soft-contact samples sigmoid(4-distance); original logcontact score; min/mean/max surface distance divided256; longest hardnear4 run/32. No action ID, scene ID, label or futuretruth input.',
      fitting='Train split only, fixed positive affine scalar baseline; readout logistic ridge meanBCE+.01/2*||w||^2 after train-only feature scaling. Intercept unpenalized. Uniform-query8 readout has same424labeled examples as updater; fullpool uses all1023 untried actions per53scene, explicitly different supervision amount.',
      eval='Full1023 frozenQ0 predictions; discovery12/confirmation10 exact same first-failure population and candidate indices. No memory update or changed first action. Direct attribution diagnostic, not improved worldmodel claim.',
      oracle='On same8cachedqueries compare Q0prediction+probe, observedlatent+probe, simulatortruegeometry through same scorer; diagnostic only, never used as deployment input or fitted readout features.',
      limitations=['One trainedQ0 andfold','Evaluation train-template scenes previouslyexamined','Fullpool readout has much more label supervision than updater/query8, not fair method-only comparison','New readout can improve actionselection without any improvement in physics prediction'])
    assert not (RR/'PLAN.json').exists();write(RR/'PLAN.json',p)


def verify():
    p=read(RR/'PLAN.json');assert p['source_sha']==sha(RR/'experiment.py') and p['parent_plan']==sha(R/'PLAN.json') and p['scorer_sha']==sha(R/'code/scorer_torch.py');return checked()


def collect(shard,nshards):
    import torch
    from scorer_torch import distances,scores_from_distances
    from fast_inference import PreparedScorer
    from common import surface_dist
    from attribution_a_metrics import log_score
    plan=verify();team=team_new();team.model.requires_grad_(False);scorer=PreparedScorer(team.head,team.rule)
    try:
        for task in (plan['fit']+plan['discovery']+plan['confirmation'])[shard::nshards]:
            out=RR/'data'/task.replace(':','_')
            if (out/'complete.json').exists():assert read(out/'complete.json')['plan']==sha(RR/'PLAN.json');continue
            row=data(task)
            if not row['failed']:
                write(out/'complete.json',dict(complete=True,plan=sha(RR/'PLAN.json'),failed=False));continue
            team.load(task);dd=[];scores=[];old=np.load(A/'cache'/task.replace(':','_')/'q0/0/scores.npy')
            with torch.no_grad():
                for start in range(0,1024,32):
                    z,a=team.chunk(start)
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a)
                    fs=[f[None] for f in team.pool['features'][start:start+32]]
                    scores.append(scorer(pred,fs).cpu().numpy());dd.append(distances(team.head,pred,fs).cpu().numpy())
                full=np.concatenate(scores);np.testing.assert_array_equal(full,old);dd=np.concatenate(dd)
                reconstructed=log_score(dd,team.rule);error=float(abs(reconstructed-full).max());assert error<5e-4,(task,error)
                observed=torch.from_numpy(row['target']).cuda().float()
                oracle_latent=scorer(observed,[f[:1] for f in row['features']]).cpu().numpy()
                true_dist=np.stack([surface_dist(f) for f in row['features']]);oracle_geometry=log_score(true_dist,team.rule)
            ar.lr.source().data.save_npz(out/'features.npz',distance=dd,old_score=old,labels=team.pool['status']>0,indices=team.pool['action_idx'],query=np.array(row['query']),first=np.array(row['first']),oracle_latent=oracle_latent,oracle_geometry=oracle_geometry)
            write(out/'complete.json',dict(complete=True,plan=sha(RR/'PLAN.json'),failed=True,score_maxerr=error,exact_original_scores=True,sha256=sha(out/'features.npz')))
            print('READOUT_DATA',task,error,flush=True)
    finally:ar.lr.close(team)


def design(d,score):
    from scipy.special import expit
    p=expit(4-d.astype(float));hard=d<4;run=np.zeros(len(d));longest=np.zeros(len(d))
    for col in hard.T:
        run=(run+1)*col;longest=np.maximum(run,longest)
    return np.column_stack([p,score,d.min(1)/256,d.mean(1)/256,d.max(1)/256,longest/32])


def metrics(s,y):
    order=np.argsort(-s,kind='stable');pos=s[y];neg=np.sort(s[~y])
    auc=float((np.searchsorted(neg,pos,'left')+np.searchsorted(neg,pos,'right')).sum()/(2*len(pos)*len(neg))) if len(pos) and len(neg) else None
    return {**{f'top{k}':int(y[order[:k]].any()) for k in [1,3,5,8]},'auc':auc}


def fit():
    assert read(RR/'FIT_V2.json')['source_sha256']==sha(Path(__file__))
    from scipy.optimize import minimize
    from scipy.special import expit
    plan=verify();rows={}
    for task in plan['fit']+plan['discovery']+plan['confirmation']:
        d=RR/'data'/task.replace(':','_');r=read(d/'complete.json');assert r['complete'] and r['plan']==sha(RR/'PLAN.json')
        if not r['failed']:continue
        assert r['sha256']==sha(d/'features.npz')
        with np.load(d/'features.npz') as z:rows[task]={k:z[k] for k in z.files}
        rows[task]['x']=design(rows[task]['distance'],rows[task]['old_score'])
    models={};cal=read(R/'DATA_AND_CALIBRATION.json')
    for arm in ['query8_readout','fullpool_readout']:
        xs=[];ys=[]
        for t in plan['fit']:
            if t not in rows:continue
            r=rows[t];idx=r['query'] if arm=='query8_readout' else np.flatnonzero(np.arange(1024)!=r['first'])
            xs.append(r['x'][idx]);ys.append(r['labels'][idx])
        x=np.concatenate(xs);y=np.concatenate(ys).astype(float);mean=x.mean(0);std=np.maximum(x.std(0),1e-5);x=(x-mean)/std
        def f(v):
            logits=x@v[:-1]+v[-1];error=expit(logits)-y
            loss=np.mean(np.logaddexp(0,logits)-y*logits)+.005*np.square(v[:-1]).sum()
            g=np.r_[x.T@error/len(y)+.01*v[:-1],error.mean()]
            return float(loss),g
        start=np.zeros(x.shape[1]+1);start[-1]=np.log((y.mean()+1e-6)/(1-y.mean()+1e-6))
        result=minimize(f,start,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8});assert result.success,result.message
        models[arm]=dict(mean=mean.tolist(),std=std.tolist(),weight=result.x[:-1].tolist(),bias=float(result.x[-1]),train_examples=len(y),train_positives=int(y.sum()),objective=float(result.fun),iterations=int(result.nit))
    write(RR/'MODELS.json',models);reports={}
    for split in ['fit','discovery','confirmation']:
        task_metrics={};oracle={}
        for task in plan[split]:
            if task not in rows:continue
            r=rows[task];keep=np.arange(1024)!=r['first'];y=r['labels'][keep];score_arrays={'original':r['old_score'],'scalar_calibration':r['old_score'].copy()}
            assert np.array_equal(np.argsort(-score_arrays['original'],kind='stable'),np.argsort(-score_arrays['scalar_calibration'],kind='stable'))
            for arm,m in models.items():score_arrays[arm]=((r['x']-m['mean'])/m['std'])@m['weight']+m['bias']
            task_metrics[task]={name:metrics(v[keep],y) for name,v in score_arrays.items()}
            qi=r['query'];qy=r['labels'][qi];oracle[task]={name:metrics(v,qy) for name,v in [('predicted',r['old_score'][qi]),('observed_latent',r['oracle_latent']),('true_geometry',r['oracle_geometry'])]}
            ar.lr.source().data.save_npz(RR/'evaluation'/split/task.replace(':','_')/'scores.npz',**score_arrays,labels=r['labels'],indices=r['indices'],first=r['first'])
        def aggregate(ms):
            names=list(next(iter(ms.values())))
            return {name:{k:float(np.mean([r[name][k] for r in ms.values() if r[name][k] is not None])) if any(r[name][k] is not None for r in ms.values()) else None for k in ['top1','top3','top5','top8','auc']} for name in names}
        reports[split]=dict(n=len(task_metrics),summary=aggregate(task_metrics),per_task=task_metrics,oracle_query_summary=aggregate(oracle),oracle_defined_auc_scenes=sum(v['predicted']['auc'] is not None for v in oracle.values()),oracle_per_task=oracle)
    write(RR/'SUMMARY.json',dict(complete=True,plan=sha(RR/'PLAN.json'),models=models,reports=reports,ranking_data_source='Frozen Q0 predictions/probe; no updater or latent edits; only success readout trained'))
    print('READOUT_COMPLETE',{s:r['summary'] for s,r in reports.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','collect','fit']);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=8);a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='collect':collect(a.shard,a.nshards)
    else:fit()
