from study import *
import torch
from torch.utils.checkpoint import checkpoint
from scorer_torch import score,supervised_terms
from memory_geometry_followup import geometry_loss


def tensor(v):return torch.from_numpy(np.asarray(v).copy()).cuda().float()


def state_for(team,row):
    state=team.model.initial_state()
    residual=tensor(row['support_observed']-row['support_prediction'])[None]
    action=torch.tensor([row['support_action']],device='cuda',dtype=torch.float32)
    with torch.autocast('cuda',dtype=torch.bfloat16):return team.model.update_state(state,residual,action)


def terms(team,row,state,cal,grad=True):
    z=tensor(row['z0']);a=tensor(row['actions'])
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a,state.expand(len(a),-1,-1)).float()
    target=tensor(row['target']);l1=(pred[:,1:]-target[:,1:]).abs().mean()
    geo=geometry_loss(team.probe,pred,row['features'],[row['times']]*len(a))
    def score_fn(p):return score(team.head,p,row['features'])
    scores=checkpoint(score_fn,pred,use_reentrant=False) if grad else score_fn(pred)
    bce,pair=supervised_terms(scores,row['labels'],cal)
    return {'prediction':l1+.1*geo,'l1':l1,'geometry':geo,'bce':bce,'pair':pair},pred,scores


def smoke():
    from fast_inference import PreparedScorer
    plan=checked();cal=read(R/'DATA_AND_CALIBRATION.json');team=team_new();model=team.model
    rows=[data(t) for t in plan['fit']];rows=[r for r in rows if r['failed']]
    mixed=[r for r in rows if 0<sum(r['labels'])<8];assert mixed
    chosen=mixed[:8];records=[];ratios={'bce':[],'pair':[]};reader=ar.lr.source().old.reader_hash(model);init=ar.lr.source().old.ident(model)
    start=time.time()
    try:
        for row in chosen:
            state=model.initial_state().detach().clone().requires_grad_(True)
            losses,pred,scores=terms(team,row,state,cal)
            reference=PreparedScorer(team.head,team.rule)(pred.detach(),[f[:1] for f in row['features']])
            error=float((scores.detach()-reference).abs().max());assert error<5e-4,(row['task'],error)
            norms={}
            for loss in ['prediction','bce','pair']:
                g=torch.autograd.grad(losses[loss],state,retain_graph=True)[0]
                assert torch.isfinite(g).all() and g.norm()>0,(loss,row['task'])
                norms[loss]=float(g.norm())
            for loss in ratios:ratios[loss].append(norms['prediction']/norms[loss])
            records.append(dict(task=row['task'],positives=sum(row['labels']),score_maxerr=error,norms=norms,losses={k:float(v.detach()) for k,v in losses.items()}))
            del state,losses,pred,scores
        weights={k:float(np.median(v)) for k,v in ratios.items()}
        # Real updater backward (initial identity has expected zero upstream grads).
        row=chosen[0];state=state_for(team,row);losses,pred,scores=terms(team,row,state,cal)
        full=losses['prediction']+weights['bce']*losses['bce'];full.backward()
        grad={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
        assert any(v>0 for v in grad.values()) and all(not n.startswith('base.') and n!='initial_tokens' for n in grad)
        assert all(p.grad is None for h in [team.probe,team.head] for p in h.parameters())
        assert reader==ar.lr.source().old.reader_hash(model) and init==ar.lr.source().old.ident(model)
        # Changing future trajectory metadata cannot change deployed scores.
        altered=row['features'].copy();altered[:,1:]=123.
        with torch.no_grad():
            before=score(team.head,pred.detach(),row['features']);after=score(team.head,pred.detach(),altered)
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        # Scalar probe-input direction finite difference at frozen prediction.
        detached=pred.detach().requires_grad_(True);ss=score(team.head,detached,row['features']);g=torch.autograd.grad(ss.mean(),detached)[0]
        direction=g/g.norm();eps=.05
        with torch.no_grad():fd=float((score(team.head,detached+eps*direction,row['features']).mean()-score(team.head,detached-eps*direction,row['features']).mean())/(2*eps))
        analytical=float((g*direction).sum());relative=abs(fd-analytical)/max(abs(analytical),1e-12)
        assert np.isfinite(fd) and fd>0 and relative<.15,(fd,analytical,relative)
        write(R/'SMOKE_AND_WEIGHTS.json',dict(passed=True,plan=sha(R/'PLAN.json'),weights=weights,calibration_tasks=[r['task'] for r in chosen],calibration_scope='FIT mixed-label query scenes only; both supervised terms matched to prediction gradient norm at initial memory state; no validation labels',records=records,initial_sha=init,reader_sha=reader,parameter_gradients=grad,finite_difference=dict(value=fd,analytical=analytical,relative=relative),future_geometry_input_invariant=True,seconds=time.time()-start))
        print('SMOKE_PASSED',weights,relative,time.time()-start,flush=True)
    finally:ar.lr.close(team)


def run(index,seed=None):
    plan=checked();seed=plan['seed'] if seed is None else seed;arm=ARMS[index];loss_kind,condition=arm.split('_');folder=R/'models'/f'seed{seed}'/arm
    weights=read(R/'SMOKE_AND_WEIGHTS.json');assert weights['passed'];cal=read(R/'DATA_AND_CALIBRATION.json')
    assert not (folder/'complete.json').exists() and not (folder/'progress.json').exists(),'No silent restart of trained or partial arm'
    team=team_new(seed,condition);model=team.model;reader=ar.lr.source().old.reader_hash(model);init=ar.lr.source().old.ident(model)
    rows=[data(t) for t in plan['fit']];rows=[r for r in rows if r['failed']]
    order=[];rng=np.random.default_rng([20261021,seed]);start=time.time();events=[]
    while len(order)<plan['steps']:order+=rng.permutation(len(rows)).tolist()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=.05)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,plan['steps']);params=[p for p in model.parameters() if p.requires_grad]
    try:
        for step,idx in enumerate(order[:plan['steps']]):
            row=rows[idx];opt.zero_grad(set_to_none=True);state=state_for(team,row)
            losses,pred,scores=terms(team,row,state,cal);loss=losses['prediction']
            if loss_kind!='prediction':loss=loss+weights['weights'][loss_kind]*losses[loss_kind]
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
            opt.step();sched.step()
            events.append(dict(step=step+1,task=row['task'],positive=sum(row['labels']),mixed=0<sum(row['labels'])<8,loss=float(loss.detach()),grad_norm=float(norm),**{k:float(v.detach()) for k,v in losses.items()}))
            del loss,losses,pred,scores,state
            if (step+1)%8==0:
                write(folder/'progress.json',dict(arm=arm,seed=seed,steps=step+1,seconds=time.time()-start,last=events[-1]));print('TRAIN',arm,step+1,events[-1],flush=True)
        assert reader==ar.lr.source().old.reader_hash(model)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        folder.mkdir(parents=True,exist_ok=True);path=folder/'updater.pt';tmp=folder/'updater.tmp'
        trainable={n for n,p in model.named_parameters() if p.requires_grad}
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k in trainable},tmp);tmp.replace(path)
        write(folder/'events.json',events)
        write(folder/'complete.json',dict(complete=True,plan=sha(R/'PLAN.json'),arm=arm,seed=seed,steps=plan['steps'],checkpoint_sha=sha(path),initial_sha=init,reader_sha=reader,query_schedule_sha=hashlib.sha256(json.dumps([(e['task'],e['positive']) for e in events]).encode()).hexdigest(),active_pair_steps=sum(e['mixed'] for e in events),seconds=time.time()-start))
    finally:ar.lr.close(team)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train']);p.add_argument('--index',type=int,default=0);p.add_argument('--seed',type=int);a=p.parse_args();smoke() if a.mode=='smoke' else run(a.index,a.seed)
