"""Read-only, fixed-history interventions on the trained Q0-first residual updater.

No training, encoder service, or new simulations. Only saved canonical truth.
The zero intervention uses the RESIDUAL-TRAINED model, not the trained zero arm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

P = Path('/path/to/phyre-unc')
SOURCE = P/'results/q0_first_feedback_20261018'
OUT = P/'results/residual_fusion_diagnostic_20261019'
sys.path[:0] = [str(SOURCE/'code'), str(P/'pipeline')]


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)


def select_tasks(plan,panel,n):
    # Only scenes with a failed first action have an update to inspect.
    # Hash order within template, round robin templates; never select by later gain/loss.
    groups={}
    for task in plan['panels'][panel]['tasks']:
        row=read(SOURCE/'evaluation'/panel/'residual'/task.replace(':','_')/'result.json')
        if len(row['events']) < 2: continue
        assert not row['events'][0]['solved']
        groups.setdefault(task.split(':')[0],[]).append(task)
    for ts in groups.values():ts.sort(key=lambda t:hashlib.sha256(('fusion-v1:'+t).encode()).hexdigest())
    chosen=[]
    while len(chosen)<n and any(groups.values()):
        for template in sorted(groups,key=lambda t:hashlib.sha256(('fusion-template-v1:'+t).encode()).hexdigest()):
            if groups[template]:chosen.append(groups[template].pop(0))
            if len(chosen)==n:break
    return chosen


def register():
    plan=read(SOURCE/'PLAN.json')
    target={'source_plan_sha256':sha(SOURCE/'PLAN.json'),
        'source_summary_sha256':sha(SOURCE/'SUMMARY.json'),
        'diagnostic_code_sha256':sha(__file__),
        'tasks':{panel:select_tasks(plan,panel,10) for panel in ['cross','within']},
        'stages':'First failed update and fourth failed update, or latest available earlier update; duplicate removed',
        'selection':'SHA256 template order, round robin, SHA256 scene order, first-failure eligibility only; no later outcome selection',
        'variants':['real','zero_residual','shuffle_pooled_tokens','scale_0.1','scale_10','no_action_embedding'],
        'conditioning':'Identical current memory, attempted action, prior real-residual action history, and tried mask',
        'observations':'Read existing canonical cache only; missing cache blocks case, no encoder/simulation fallback',
        'ranking':'All1024 candidates, original32batch BF16 prediction/FP32 probe/float64 score',
        'limits':['Small observational sample; not a new end-to-end S@9 evaluation',
                  'Zero residual is an intervention on residual-trained model, not the trained zero-residual control',
                  'Interventions can be out of distribution; no architecture superiority claim',
                  'Pooling magnitude/cancellation does not measure task-relevant information loss',
                  'Scales and normalization diagnostics do not alone establish failure causality']}
    path=OUT/'PLAN.json'
    if path.exists():assert read(path)==target
    else:write(path,target)
    print(json.dumps(target['tasks']),flush=True)


def cached_truth(team,task,j):
    import numpy as np
    key=f'{task.replace(":","_")}_{int(team.pool["action_idx"][j]):06d}'
    root=team.truth.root
    path=root/str(int(hashlib.sha256(key.encode()).hexdigest(),16)%32)/(key+'.npz')
    if not path.exists():raise FileNotFoundError(f'Canonical truth unavailable: {path}')
    with np.load(path) as z:
        assert str(z['task'])==task
        assert str(z['plan'])==sha(team.data_source_plan)
        assert int(z['status'])==int(team.pool['status'][j])
        np.testing.assert_array_equal(z['action'],team.pool['simulation_action'][j])
        latent=z['latent'].copy()
    return latent,sha(path)


def ranking_metrics(scores,labels,tried,reference=None):
    import numpy as np
    ids=np.flatnonzero(~tried);order=ids[np.argsort(-scores[ids],kind='stable')];hits=np.flatnonzero(labels[order])
    out={'next_local':int(order[0]),'next_success':bool(labels[order[0]]),
         'remaining_positive':int(labels[ids].sum()),'best_success_rank':int(hits[0]+1) if len(hits) else None,
         'success_in_top8':bool(len(hits) and hits[0]<8),'top10':order[:10].tolist()}
    pos=scores[ids][labels[ids]];neg=scores[ids][~labels[ids]]
    out['pair_auc']=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))) if len(pos) and len(neg) else None
    if reference is not None:
        reforder=ids[np.argsort(-reference[ids],kind='stable')]
        out.update(top1_changed=bool(order[0]!=reforder[0]),top10_overlap=float(len(set(order[:10])&set(reforder[:10]))/min(10,len(ids))),
                   score_rms_difference=float(np.sqrt(np.mean((scores[ids]-reference[ids])**2))))
    return out


def metric_tensor(x):
    import torch
    a=x.detach().float();row=a.square().mean(-1).sqrt().flatten()
    centered=a-a.mean(-1,keepdim=True)
    return {'rms':float(a.square().mean().sqrt()),'centered_rms':float(centered.square().mean().sqrt()),
            'token_rms_median':float(row.median()),'token_rms_p90':float(torch.quantile(row,.9))}


def compare(x,ref):
    import torch
    a=x.detach().float();b=ref.detach().float();diff=a-b
    return {'rms_difference':float(diff.square().mean().sqrt()),
            'relative_l2':float(diff.norm()/b.norm().clamp_min(1e-12)),
            'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))}


def update_parts(model,state,residual,action,variant,seed):
    import torch
    from torch.nn import functional as F
    b,t,s,dim=residual.shape;g=int(round(s**.5));assert g*g==s and t%model.mem_steps==0
    sampled=residual[:,::t//model.mem_steps]
    grid=sampled.reshape(b*model.mem_steps,g,g,dim).permute(0,3,1,2)
    pooled=F.adaptive_avg_pool2d(grid,model.mem_grid)
    pooled_abs=F.adaptive_avg_pool2d(grid.abs(),model.mem_grid)
    z=pooled.permute(0,2,3,1).reshape(b,model.mem_steps,model.mem_grid**2,dim)
    original_pooled=z
    if variant=='zero_residual':z=torch.zeros_like(z)
    elif variant=='shuffle_pooled_tokens':
        gen=torch.Generator(device='cpu').manual_seed(seed)
        idx=torch.randperm(z.shape[1]*z.shape[2],generator=gen).to(z.device)
        z=z.flatten(1,2)[:,idx].reshape_as(z)
    elif variant.startswith('scale_'):z=z*float(variant.split('_')[1])
    with torch.autocast(state.device.type,dtype=torch.bfloat16,enabled=state.device.type=='cuda'):
        rp=model.residual_in(z)
        ae=model.action_in(action)[:,None,None]
        if variant=='no_action_embedding':ae=torch.zeros_like(ae)
        mixed=rp+model.update_time+model.update_space+ae
        obs=model.observation_norm(mixed.flatten(1,2))
        u=model.update_attention(model.state_norm(state),obs,obs,need_weights=False)[0]
        intermediate=state+u
        updated=intermediate+model.update_ff(model.ff_norm(intermediate))
        bias=model.residual_in(torch.zeros_like(z))
    stats={'raw_residual':metric_tensor(residual),'time_sampled':metric_tensor(sampled),
        'pooled':metric_tensor(original_pooled),'residual_signal':metric_tensor(rp.float()-bias.float()),
        'residual_projection_with_bias':metric_tensor(rp),'residual_bias':metric_tensor(bias),
        'action_embedding':metric_tensor(ae),'time_embedding':metric_tensor(model.update_time),
        'space_embedding':metric_tensor(model.update_space),
        'nonresidual_sum':metric_tensor(bias+model.update_time+model.update_space+ae),
        'mixed_before_LN':metric_tensor(mixed),'after_LN':metric_tensor(obs),
        'attention_delta':metric_tensor(u),'full_update':metric_tensor(updated-state),
        'spatial_abs_retention':float(pooled.abs().mean()/pooled_abs.mean().clamp_min(1e-12)),
        'temporal_sample_rms_ratio':float(sampled.square().mean().sqrt()/residual.square().mean().sqrt().clamp_min(1e-12))}
    return updated,mixed,obs,stats


def run(panel,shard,nshards):
    import numpy as np
    import torch
    import selector_feedback_run as src
    torch.set_num_threads(2)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert 'DISABLE_ADDMM_CUDA_LT' not in os.environ
    manifest=read(OUT/'PLAN.json');assert sha(__file__)==manifest['diagnostic_code_sha256']
    assert sha(SOURCE/'PLAN.json')==manifest['source_plan_sha256']
    checkpoint=read(SOURCE/'models'/panel/'residual'/'complete.json')
    assert sha(checkpoint['checkpoint'])==checkpoint['checkpoint_sha256']
    plan=src.data.checked();team=src.Team(panel,'residual',plan)
    team.data_source_plan=src.data.SOURCE_R/'PLAN.json'
    team.model.load_state_dict(torch.load(checkpoint['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
    team.model.eval().requires_grad_(False);frozen=src.old.ident(team.model)
    assert not team.model.zero_residual
    for task in manifest['tasks'][panel][shard::nshards]:
        dest=OUT/'cases'/panel/task.replace(':','_')/'result.json'
        if dest.exists():assert read(dest)['plan_sha256']==sha(OUT/'PLAN.json');continue
        start=time.time();team.load(task)
        folder=SOURCE/'evaluation'/panel/'residual'/task.replace(':','_')
        row=read(folder/'result.json')
        with np.load(folder/'scores.npz') as z:
            scores=z['scores'].copy();selected=z['selected'].copy()
            np.testing.assert_array_equal(z['indices'],team.pool['action_idx'])
            np.testing.assert_array_equal(z['labels'],team.pool['status']>0)
        stages=sorted(set([0,min(3,len(selected)-2)]));state=team.model.initial_state().detach().clone()
        tried=np.zeros(1024,bool);cases=[];cache_hashes={}
        with torch.no_grad():
            for turn in range(max(stages)+1):
                j=int(selected[turn]);assert not row['events'][turn]['solved'];tried[j]=True
                truth,cache_hash=cached_truth(team,task,j);cache_hashes[str(j)]=cache_hash
                truth=torch.from_numpy(truth[None]).cuda().float()
                pred=team.predict(state,j)[j%32:j%32+1]
                residual=truth-pred;action=torch.from_numpy(team.pool['action'][j:j+1]).cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):actual=team.model.update_state(state,residual,action)
                if turn not in stages:state=actual;continue
                before=team.scores(state)
                np.testing.assert_array_equal(before,scores[turn])
                seed=int(hashlib.sha256(f'{panel}:{task}:{turn}'.encode()).hexdigest()[:8],16)
                real,mix0,obs0,stats0=update_parts(team.model,state,residual,action,'real',seed)
                assert torch.equal(real,actual),'Instrumented updater differs from actual'
                real_scores=team.scores(real)
                np.testing.assert_array_equal(real_scores,scores[turn+1])
                variant_rows={};raw_scores={'before':before,'real':real_scores}
                for variant in manifest['variants']:
                    if variant=='real':out,mix,obs,stats=real,mix0,obs0,stats0;rank=real_scores
                    else:
                        out,mix,obs,stats=update_parts(team.model,state,residual,action,variant,seed)
                        rank=team.scores(out)
                    variant_rows[variant]={'components':stats,'mixed_change_vs_real':compare(mix,mix0),
                        'normalized_change_vs_real':compare(obs,obs0),'memory_change_vs_real':compare(out,real),
                        'update_change_vs_real':compare(out-state,real-state),
                        'memory_change_vs_before':compare(out,state),
                        'ranking':ranking_metrics(rank,team.pool['status']>0,tried,real_scores)}
                    raw_scores[variant]=rank
                sf=dest.parent/f'update_{turn+1}_scores.npz';sf.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(sf,**raw_scores,indices=team.pool['action_idx'],labels=team.pool['status']>0,tried=tried)
                cases.append({'update_number':turn+1,'attempted_local':j,'attempted_action_idx':int(team.pool['action_idx'][j]),
                    'attempted_action':team.pool['action'][j].tolist(),'before':ranking_metrics(before,team.pool['status']>0,tried),
                    'variants':variant_rows,'instrumentation_exact':True,'original_before_after_scores_exact':True,
                    'residual_shape':list(residual.shape),'pooled_shape':[1,team.model.mem_steps,team.model.mem_grid**2,residual.shape[-1]]})
                state=actual
                print('CASE',panel,task,'update',turn+1,'elapsed_s',round(time.time()-start,1),flush=True)
        assert src.old.ident(team.model)==frozen
        write(dest,{'complete':True,'task':task,'panel':panel,'plan_sha256':sha(OUT/'PLAN.json'),
            'checkpoint_sha256':checkpoint['checkpoint_sha256'],'gpu':torch.cuda.get_device_name(0),
            'cache_hashes':cache_hashes,'seconds':time.time()-start,'cases':cases})
    team.truth.close()
    write(OUT/'shards'/f'{panel}_{shard}.json',{'complete':True,'plan_sha256':sha(OUT/'PLAN.json')})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','run']);ap.add_argument('--panel',choices=['cross','within']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=1);a=ap.parse_args()
    register() if a.mode=='register' else run(a.panel,a.shard,a.nshards)
