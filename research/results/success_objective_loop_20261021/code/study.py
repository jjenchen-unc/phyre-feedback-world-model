"""Success supervision diagnosis: matched single-feedback updater experiments."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[3];R=P/'results/success_objective_loop_20261021';A=P/'results/current_probe_adaptation_20261020';D=P/'results/next_direction_loop_20261020'
sys.path.insert(0,str(A/'code'))
import adapt_runtime as ar
read,write,sha=ar.read,ar.write,ar.sha
ARMS=[f'{loss}_{condition}' for loss in ['prediction','bce','pair'] for condition in ['residual','zero']]


def checked():
    p=read(R/'PLAN.json')
    for path,h in p['dependencies'].items():assert sha(path)==h,path
    return p


def register():
    prior=ar.checked();d=read(D/'PLAN.json');assert read(D/'COMPLETION_AUDIT.json')['passed']
    dependencies={str(p):sha(p) for p in sorted((R/'code').glob('*.py'))}
    dependencies.update(prior['dependencies']);dependencies[str(A/'PLAN.json')]=sha(A/'PLAN.json')
    dependencies[str(D/'PLAN.json')]=sha(D/'PLAN.json')
    plan=dict(fit=prior['fit'],discovery=d['discovery'],confirmation=d['confirmation'],dependencies=dependencies,
      arms=ARMS,seed=7919,steps=128,lr=3e-4,geometry_weight=.1,latent_loss='L1',batch_scenes=1,
      deadline='2026-10-21 09:00 America/New_York',
      hypothesis='Does explicit success supervision improve early action ranking, and does its gain require the actual feedback residual?',
      round0='Verify differentiable scorer against exact deployment scorer; cached actual first-Q0-failure support, uniform8 untried queries. Training-only positive-slope logistic calibration on full Q0 pools, same fixed score scale for BCE and pairwise. Measure gradients and effective positive-pair coverage before training.',
      round1='Fresh updater x6: prediction/BCE/pairwise by centered-real/zero residual. Same Q0, fixed initial tokens/geometry probe/ranking probe/scorer, same scene/query order/optimizer steps. No near filters, label-based support selection, extra ranking queries or reuse of trained updater. Only one update: pre-update Q0 is identical, so its cached prediction is exact for every arm.',
      loss='L1future +0.1geometry plus lambda*BCE or lambda*mean all positive-negative softplus on identical8 queries. Lambda training-calibrated to median ratio of prediction versus supervised gradient norms at memory state; fixed afterwards. Pair term zero if no positive-negative pair, retain prediction loss and report coverage.',
      validation='Full1024 original scorer, same first failed trial and candidate pool, next top1/3/5/8 and scene AUC, Q0 comparator. All6 on discovery16 then confirmation16 (no per-arm confirmation suppression). Query geometry/latent errors secondary. Finite small development screen, not deployment multi-turn or unseen templates.',
      iteration='After round1 inspect paired gains vs own prediction control AND trained zero control on both partitions. If promising replicate baseline+winning loss real/zero with fresh second updater seed. If success supervision ineffective diagnose score calibration/saturation, positive coverage and readout bottleneck before bounded second experiment. Up to12 total new updater fits; no massive full pretraining.',
      readout_followup='If scorer bottleneck implicated, fit small success readout on frozen predicted goal-distance features with train-only scaling and compare same unmodified latent/probe. Must distinguish this from updater supervision.',
      limitations=['One cross fold and upstream model','Previously examined training-template validation scenes, not unseen cross-template generalization','First-failure subgroup; report denominator and include initial successes separately','Natural uniform8 may rarely contain both labels; loss/data interaction measured, not erased','Single-feedback screening does not establish multi-turn benefit','Residual centering fixed in all real-input arms; do not conflate with new loss','Success labels add supervision beyond original latent/geometry objective'])
    assert not (R/'PLAN.json').exists();write(R/'PLAN.json',plan)


def team_new(seed=7919,condition='residual'):
    import torch,copy
    team=ar.make_team();src=ar.lr.source()
    # ar.make_team has the original Q0 reader plus fresh updater; always reconstruct
    # under explicit seed, copying only base and initial tokens.
    torch.manual_seed(seed)
    model=src.old.b.PrefixPredictor(copy.deepcopy(team.model.base),team.model.base.inp[1].in_features,stage='updater',checkpoint_blocks=True).cuda()
    with torch.no_grad():model.initial_tokens.copy_(team.model.initial_tokens)
    assert src.old.reader_hash(model)==src.old.reader_hash(team.model)
    team.center_hook.remove();team.model=model.eval();team.input_mode='centered' if condition=='residual' else 'zero'
    team.center_hook=model.residual_in.register_forward_pre_hook(lambda module,args:(ar.lr.ct.transform(args[0],team.input_mode),))
    return team


def strict_observation(team,j):
    key=f'{team.task.replace(":","_")}_{int(team.pool["action_idx"][j]):06d}';bucket=str(int(hashlib.sha256(key.encode()).hexdigest(),16)%32)
    roots=[team.truth.root]+team.truth.parents
    for root in roots:
        path=Path(root)/bucket/(key+'.npz')
        if not path.exists():continue
        with np.load(path) as z:
            assert str(z['task'])==team.task and int(z['status'])==int(team.pool['status'][j])
            np.testing.assert_array_equal(z['action'],team.pool['simulation_action'][j])
            return {k:z[k].copy() for k in ['latent','features']},path
    raise FileNotFoundError(('Cached observation missing; no simulator/encoder fallback permitted',team.task,j))


def pack(shard,nshards):
    import torch
    from fast_inference import PreparedScorer
    from scorer_torch import score
    plan=checked();team=team_new();tasks=(plan['fit']+plan['discovery']+plan['confirmation'])[shard::nshards]
    try:
        for task in tasks:
            out=R/'data'/task.replace(':','_')
            if (out/'complete.json').exists():assert read(out/'complete.json')['plan']==sha(R/'PLAN.json');continue
            team.load(task);old=np.load(A/'cache'/task.replace(':','_')/'q0/0/scores.npy');j=int(old.argmax());failed=bool(team.pool['status'][j]<0)
            if not failed:
                write(out/'complete.json',dict(complete=True,plan=sha(R/'PLAN.json'),task=task,first=j,failed=False));continue
            meta=read(A/'cache'/task.replace(':','_')/'smooth_centered/1/metadata.json');assert meta['tried']==[j];qi=meta['query'];assert j not in qi
            support,path=strict_observation(team,j);targets=[];features=[];sources=[str(path)]
            for q in qi:
                obs,path=strict_observation(team,q);targets.append(obs['latent']);features.append(obs['features']);sources.append(str(path))
            with torch.no_grad():
                z,a=team.chunk(j//32*32)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a)
                fs=[f[None] for f in team.pool['features'][j//32*32:j//32*32+32]]
                exact=PreparedScorer(team.head,team.rule)(pred,fs).cpu().numpy();np.testing.assert_array_equal(exact,old[j//32*32:j//32*32+32])
                differentiable=score(team.head,pred,fs).cpu().numpy();err=float(np.max(np.abs(exact-differentiable)));assert err<5e-4,(task,err)
                agreement=bool(np.array_equal(np.argsort(-exact,kind='stable'),np.argsort(-differentiable,kind='stable')))
                support_prediction=pred[j%32].float().cpu().numpy()
            ar.save_array(out/'support_prediction.npy',support_prediction)
            ar.save_array(out/'support_observed.npy',support['latent'].astype(np.float32))
            ar.save_array(out/'target.npy',np.stack(targets).astype(np.float32))
            ar.save_array(out/'features.npy',np.stack(features))
            ar.save_array(out/'z0.npy',team.pool['z0'][qi])
            ar.save_array(out/'actions.npy',team.pool['action'][qi])
            write(out/'complete.json',dict(complete=True,plan=sha(R/'PLAN.json'),task=task,first=j,failed=True,query=qi,times=meta['times'],labels=(team.pool['status'][qi]>0).tolist(),support_action=team.pool['action'][j].tolist(),sources=sources,torch_score_maxerr=err,chunk_order_equal=agreement,reader_sha=ar.lr.source().old.reader_hash(team.model),hashes={p.name:sha(p) for p in out.glob('*.npy')}))
            print('PACK',task,'positive',sum(team.pool['status'][qi]>0),'score_error',err,flush=True)
    finally:ar.lr.close(team)


def data(task):
    d=R/'data'/task.replace(':','_');m=read(d/'complete.json');assert m['complete'] and m['plan']==sha(R/'PLAN.json')
    if m['failed']:
        m.update({k:np.load(d/(k+'.npy')) for k in ['support_prediction','support_observed','target','features','z0','actions']})
    return m


def calibration():
    from scipy.optimize import minimize
    plan=checked();summaries={};xs=[];ys=[]
    for split in ['fit','discovery','confirmation']:
        rows=[read(R/'data'/t.replace(':','_')/'complete.json') for t in plan[split]]
        assert all(r['complete'] and r['plan']==sha(R/'PLAN.json') for r in rows)
        active=[r for r in rows if r['failed']]
        summaries[split]=dict(n=len(rows),active=len(active),queries=8*len(active),positives=sum(sum(r['labels']) for r in active),mixed_label_scenes=sum(0<sum(r['labels'])<8 for r in active),torch_score_maxerr=max(r['torch_score_maxerr'] for r in active),all_chunk_orders_equal=all(r['chunk_order_equal'] for r in active))
        if split!='fit':continue
        for r in active:
            task=r['task'];s=np.load(A/'cache'/task.replace(':','_')/'q0/0/scores.npy')
            with np.load(A/'pools'/task.replace(':','_')/'pool.npz') as z:y=z['status']>0
            keep=np.arange(1024)!=r['first'];xs.append(s[keep]);ys.append(y[keep])
    x=np.concatenate(xs);y=np.concatenate(ys).astype(float);std=max(float(x.std()),1e-6);mean=float(x.mean())
    def objective(v):
        slope=np.exp(v[0])/std;logit=slope*(x-mean)+v[1]
        return float(np.mean(np.logaddexp(0,logit)-y*logit)+1e-6*np.square(v).sum())
    fit=minimize(objective,[0.,np.log((y.mean()+1e-6)/(1-y.mean()+1e-6))],method='L-BFGS-B',bounds=[(-8,8),(-20,20)])
    assert fit.success and np.isfinite(fit.fun)
    slope=float(np.exp(fit.x[0])/std);intercept=float(fit.x[1]-slope*mean)
    write(R/'DATA_AND_CALIBRATION.json',dict(complete=True,plan=sha(R/'PLAN.json'),splits=summaries,slope=slope,intercept=intercept,fit_bce=float(fit.fun),fit_positive_rate=float(y.mean()),calibration_scope='FIT first-failure Q0 full1023 pools only; positive scalar slope preserves every rank; frozen for both BCE and pair loss',fit_scenes=len(xs)))
    print('CALIBRATION',summaries,slope,intercept,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','pack','calibration']);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=8);a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='pack':pack(a.shard,a.nshards)
    else:calibration()
