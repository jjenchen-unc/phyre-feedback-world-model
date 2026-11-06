"""Adaptive round: BCE with trainable positive-affine calibration, fixed rank rule."""
import argparse,sys,importlib.util,json,time,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import study as st
spec=importlib.util.spec_from_file_location('this_study_train',ROOT/'code/train.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)
import torch
from torch.nn import functional as F
RR=ROOT/'calibration_update';read,write,sha=st.read,st.write,st.sha
ARMS=['bcecal_residual','bcecal_zero']


def register():
    st.checked();d=read(ROOT/'DATA_AND_CALIBRATION.json');maximum=d['slope']*math.log(32)+d['intercept']
    plan=dict(parent_plan=sha(ROOT/'PLAN.json'),source_sha=sha(Path(__file__)),arms=ARMS,steps=128,seed=7919,
      trigger=dict(type='Analytical diagnostic before outcome-based choice',maximum_fixed_logit=maximum,maximum_fixed_probability=1/(1+math.exp(-maximum))),
      change='Jointly optimize positive slope exp(log_slope) and intercept for BCE. Initial mapping, fresh updater, scene/query schedule, prediction loss, frozen reader, lambda and inference geometric scoring match round1. Calibration changes loss gradients but monotone affine cannot directly alter inference order.',
      optimizer='Updater same AdamW3e-4 wd.05 clip1 cosine128; two calibration scalars Adam3e-2 wd0 cosine128, no updater clip contamination. log_slope initializedlog(.11795), intercept-2.42209 fromtrain-only fixed calibration.',
      comparison='Against original fixed-calibration BCE real/zero AND prediction real/zero on same scenes/candidates. All evaluation uses original scorer; no new deploy head.',
      controls='Calibration-only score remains positive-affine monotone, so its ranks mathematically match originalQ0; record ceiling evolution and whether gradient improvement instead sacrifices geometry.',
      limitations=['One updaterseed and128steps','Added two scalar optimizer variables; this tests mapping adaptation, not BCE vs pairwise in isolation','No claim of resolved probability calibration unless heldout calibration alsoimproves','Same previouslyexamined validation partitions'])
    assert not (RR/'PLAN.json').exists();write(RR/'PLAN.json',plan)


def checked():
    p=read(RR/'PLAN.json');assert p['source_sha']==sha(Path(__file__)) and p['parent_plan']==sha(ROOT/'PLAN.json');st.checked();return p


def train(index):
    p=checked();arm=ARMS[index];condition=arm.split('_')[1];team=st.team_new(p['seed'],condition);model=team.model;cal=read(ROOT/'DATA_AND_CALIBRATION.json');weight=read(ROOT/'SMOKE_AND_WEIGHTS.json')['weights']['bce']
    folder=RR/'models'/arm;assert not(folder/'progress.json').exists() and not(folder/'complete.json').exists()
    logslope=torch.nn.Parameter(torch.tensor(math.log(cal['slope']),device='cuda',dtype=torch.float64));bias=torch.nn.Parameter(torch.tensor(cal['intercept'],device='cuda',dtype=torch.float64))
    params=[x for x in model.parameters() if x.requires_grad];opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=.05);co=torch.optim.Adam([logslope,bias],lr=.03)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['steps']);cs=torch.optim.lr_scheduler.CosineAnnealingLR(co,p['steps'])
    rows=[st.data(t) for t in st.checked()['fit']];rows=[r for r in rows if r['failed']];rng=np.random.default_rng([20261021,p['seed']]);order=[]
    while len(order)<p['steps']:order+=rng.permutation(len(rows)).tolist()
    init=st.ar.lr.source().old.ident(model);reader=st.ar.lr.source().old.reader_hash(model);events=[];start=time.time()
    try:
        for step,idx in enumerate(order[:p['steps']]):
            row=rows[idx];opt.zero_grad(set_to_none=True);co.zero_grad(set_to_none=True);state=tr.state_for(team,row)
            losses,pred,scores=tr.terms(team,row,state,cal);logits=logslope.exp()*scores+bias;y=torch.tensor(row['labels'],device='cuda',dtype=torch.float64)
            bce=F.binary_cross_entropy_with_logits(logits,y);loss=losses['prediction']+weight*bce;assert torch.isfinite(loss);loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(gn) and torch.isfinite(logslope.grad) and torch.isfinite(bias.grad)
            opt.step();co.step();sched.step();cs.step()
            events.append(dict(step=step+1,task=row['task'],positive=sum(row['labels']),loss=float(loss.detach()),bce=float(bce.detach()),prediction=float(losses['prediction'].detach()),geometry=float(losses['geometry'].detach()),slope=float(logslope.exp().detach()),intercept=float(bias.detach()),maximum_probability=float(torch.sigmoid(logslope.exp()*math.log(32)+bias).detach()),grad_norm=float(gn)))
            del state,losses,pred,scores,logits,bce,loss
            if (step+1)%8==0:write(folder/'progress.json',events[-1]);print('CALTRAIN',arm,events[-1],flush=True)
        assert reader==st.ar.lr.source().old.reader_hash(model)
        folder.mkdir(parents=True,exist_ok=True);names={n for n,x in model.named_parameters() if x.requires_grad};path=folder/'updater.pt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k in names},path)
        write(folder/'events.json',events);write(folder/'complete.json',dict(complete=True,plan=sha(RR/'PLAN.json'),initial_sha=init,reader_sha=reader,checkpoint_sha=sha(path),steps=p['steps'],seconds=time.time()-start,slope=float(logslope.exp().detach()),intercept=float(bias.detach()),query_schedule_sha=st.hashlib.sha256(json.dumps([(e['task'],e['positive']) for e in events]).encode()).hexdigest()))
    finally:st.ar.lr.close(team)


def evaluate(stage,shard,nshards):
    from fast_inference import PreparedScorer
    p=checked();team=st.team_new();scorer=PreparedScorer(team.head,team.rule);cal=read(ROOT/'DATA_AND_CALIBRATION.json');reader=st.ar.lr.source().old.reader_hash(team.model)
    try:
        for task in st.checked()[stage][shard::nshards]:
            out=RR/'evaluation'/stage/task.replace(':','_');row=st.data(task)
            if(out/'result.json').exists():assert read(out/'result.json')['plan']==sha(RR/'PLAN.json');continue
            if not row['failed']:
                write(out/'result.json',dict(complete=True,plan=sha(RR/'PLAN.json'),first_failed=False,task=task));continue
            team.load(task);arrays={};lossrec={}
            with torch.no_grad():
                for arm in ARMS:
                    d=RR/'models'/arm;r=read(d/'complete.json');assert r['plan']==sha(RR/'PLAN.json') and r['checkpoint_sha']==sha(d/'updater.pt')
                    missing,extra=team.model.load_state_dict(torch.load(d/'updater.pt',map_location='cpu',weights_only=True),strict=False);assert not extra and all(n.startswith('base.') or n=='initial_tokens' for n in missing)
                    team.input_mode='centered' if arm.endswith('residual') else 'zero';team.model.eval().requires_grad_(False);team.model.checkpoint_blocks=False
                    state=tr.state_for(team,row);chunks=[]
                    for start in range(0,1024,32):
                        z,a=team.chunk(start)
                        with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a,state.expand(32,-1,-1))
                        chunks.append(scorer(pred,[f[None] for f in team.pool['features'][start:start+32]]).cpu().numpy())
                    arrays[arm]=np.concatenate(chunks);losses,_,scores=tr.terms(team,row,state,cal,grad=False)
                    learned=F.binary_cross_entropy_with_logits(r['slope']*scores+r['intercept'],torch.tensor(row['labels'],device='cuda',dtype=torch.float64))
                    lossrec[arm]={k:float(v) for k,v in losses.items()};lossrec[arm]['bce_learned_calibration']=float(learned)
            assert reader==st.ar.lr.source().old.reader_hash(team.model)
            st.ar.lr.source().data.save_npz(out/'scores.npz',**arrays,labels=team.pool['status']>0,indices=team.pool['action_idx'],first=np.array(row['first']))
            write(out/'result.json',dict(complete=True,plan=sha(RR/'PLAN.json'),first_failed=True,task=task,reader_unchanged=True,losses=lossrec));print('CALEVAL',stage,task,flush=True)
    finally:st.ar.lr.close(team)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','train','discovery','confirmation']);p.add_argument('--index',type=int,default=0);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=8);a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='train':train(a.index)
    else:evaluate(a.mode,a.shard,a.nshards)
