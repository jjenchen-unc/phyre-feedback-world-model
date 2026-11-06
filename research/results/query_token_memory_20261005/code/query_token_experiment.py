"""Train recurrent residual-updated tokens and test online action re-ranking."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import torch
import fresh_memory_geometry as f
from query_token_predictor import QueryTokenPredictor

FRESH=f.PROJECT/'results/memory_fresh_geometry_20261005'


def prepare(root):
    assert not (root/'manifest.json').exists();root.mkdir(parents=True,exist_ok=True)
    m=f.read(FRESH/'manifest.json');m.pop('initialization');m['scope']='Fresh recurrent query tokens; residual-only state updates; L1+.1geometry after every training prefix; bounded closed-loop pilot.'
    m['data_parent_manifest_sha256']=f.digest(FRESH/'manifest.json')
    m['initialization']='Fresh globally learned initial tokens and update/decoder weights; per-scene token state reset; no old memory checkpoint'
    m['architecture']={'n_tokens':32,'d_mem':256,'mem_steps':16,'mem_grid':10}
    m['training']['support_lengths']=[1,2,4,8];m['training']['prefix_loss']='Mean over actual prefix count; same query set at each prefix; full gradient through state recurrence'
    m['residual']='Observed latent minus frozen base prediction, with batch1 base call per observed action'
    m['closed_loop']={'budget':20,'cross_per_template':5,'within_per_template':2,'head':m['primary_head'],'update_after':'every failure before remaining attempts','parameter_gradient_at_deploy':False}
    cross=[r for r in m['final_panels'] if r['panel']=='cross_natural']
    m['closed_loop']['cross_tasks']=[r['task'] for r in f.choose_tasks(cross,5,'token_closed')]
    m['limits']+=['Recurrent architecture AND multi-prefix training differ from fresh K1 baseline; not isolated architecture attribution',
        'Training <=8 updates; closed loop can reach19, report per-update results and do not assume extrapolation works',
        'Bounded20-attempt pilot; log_area20 is NOT canonical100-attempt AUCCESS',
        'Training support sequences are uniformly sampled failures, not on-policy training trajectories']
    m['code']=f.code_hashes();f.write_json(root/'manifest.json',m)
    audit=f.read(FRESH/'DATA_AUDIT.json');assert audit['passed'] and audit['manifest_sha256']==f.digest(FRESH/'manifest.json')
    for p,h in m['source_hashes'].items():assert f.digest(p)==h
    f.write_json(root/'DATA_AUDIT.json',{'passed':True,'manifest_sha256':f.digest(root/'manifest.json'),'parent_audit_sha256':f.digest(FRESH/'DATA_AUDIT.json'),'counts':m['counts'],'near_filter_used':False})
    print('TOKEN_PREPARED',json.dumps(m['counts']),flush=True)


def models(m,seed,checkpoint=None):
    assert f.digest(m['base'])==m['base_sha256'] and f.digest(m['probe'])==m['probe_sha256']
    base,probe,rule,pc,_=f.load_models(memory=None);base.eval().requires_grad_(False);probe.eval().requires_grad_(False)
    torch.manual_seed(seed);model=QueryTokenPredictor(base,pc['Din'],**m['architecture']).cuda()
    if checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['model_type']=='query_tokens' and ck['base']==m['base'] and ck['initialization']=='fresh'
        missing,extra=model.load_state_dict(ck['state_dict'],strict=False);assert not extra and all(k.startswith('base.') for k in missing)
    heads,hh=f.load_heads(m,f.PROBES,probe);assert all(hh[h]==v for h,v in m['heads'].items())
    return base,probe,rule,model,heads


def episode(rows,seed,step):
    row,ids,times,_=f.episode(rows,seed,step)
    rng=np.random.default_rng([f.SEED,seed,step,9]);requested=int(rng.choice([1,2,4,8]))
    ctx=rng.choice(row['failures'],min(requested,len(row['failures'])),replace=False).tolist()
    return row,ids,times,ctx,requested


def save_checkpoint(folder,model,m,seed,step,prov):
    path=folder/'checkpoints'/f'step_{step:06d}.pt'
    f.atomic_torch(path,{'architecture':m['architecture'],'model_type':'query_tokens','base':m['base'],'base_sha256':m['base_sha256'],
        'state_dict':f.weights(model),'initialization':'fresh','seed':seed,'step':step,'provenance':prov})
    return path


def train(root,seed,smoke):
    m=f.read(root/'manifest.json');prov=f.provenance(root);assert m['code']==f.code_hashes()
    assert f.read(root/'DATA_AUDIT.json')['manifest_sha256']==prov['manifest']
    if not smoke:assert f.read(root/'SMOKE.json')['passed'] and f.read(root/'SMOKE.json')['provenance']==prov
    folder=root/('smoke' if smoke else 'models')/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():assert f.read(folder/'complete.json')['provenance']==prov;return
    store=f.Store();cross=f.Pack(m['cross_pack']);assert f.digest(m['cross_pack']+'_meta.npz')==m['cross_meta_sha256']
    assert __import__('hashlib').sha256('\n'.join(store.files).encode()).hexdigest()==m['store_files_sha256']
    for p,h in m['source_hashes'].items():assert f.digest(p)==h
    base,probe,rule,model,heads=models(m,seed);fixed={k:f.frozen_fingerprint(v) for k,v in [('base',base),('probe',probe)]};initial=f.frozen_fingerprint(model)
    params=[p for p in model.parameters() if p.requires_grad];cfg=m['training'];steps=8 if smoke else cfg['steps'];checks=[0,4,8] if smoke else m['checkpoints']
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay']);scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,cfg['lr'],total_steps=steps,pct_start=.5 if smoke else .03)
    rows=[r for r in m['rows'] if r['split']=='train'];start=0;draws=[];logs=[];gradient_checks={};resume=folder/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['provenance']==prov and ck['steps']==steps and ck['seed']==seed
        missing,extra=model.load_state_dict(ck['weights'],strict=False);assert not extra and all(k.startswith('base.') for k in missing)
        opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start,draws,logs,gradient_checks=ck['step'],ck['draws'],ck['logs'],ck['gradient_checks']
    if start==0:
        assert all(torch.count_nonzero(o.weight)==0 and torch.count_nonzero(o.bias)==0 for o in model.o)
        ck=save_checkpoint(folder,model,m,seed,0,prov);f.curves(root,folder,m,seed,0,ck,base,model,heads,rule,store,cross,smoke)
    t0=time.time();running=np.zeros(3);nrun=0
    for step in range(start,steps):
        row,ids,times,ctx,requested=episode(rows,seed,step)
        assert len(ctx)==len(set(ctx)) and not set(ids)&set(ctx) and not store.labels[ctx].any()
        z=torch.from_numpy(store.z(ids)).cuda().float();act=torch.from_numpy(store.actions[ids]).cuda();features=[store.feature(i) for i in ids]
        model.train();base.eval();probe.eval();state=model.initial_state();prefixes=[];opt.zero_grad(set_to_none=True)
        for j,idx in enumerate(ctx):
            observed=torch.from_numpy(store.z([idx])).cuda().float();a=torch.from_numpy(store.actions[[idx]]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                residual=model.context_residual(observed[:,None],a[:,None])[:,0].float()
                state=model.update_state(state,residual,a)
                memory,mask=model.state_memory(state,True)
                pred=model(z[:,0],act,memory.expand(len(ids),-1,-1),mask.expand(len(ids),-1)).float()
            assert torch.isfinite(state).all()
            latent=(pred[:,1:]-z[:,1:]).abs().mean();geo=f.geometry_loss(probe,pred,features,times);loss=(latent+cfg['geometry_weight']*geo)/len(ctx)
            assert torch.isfinite(loss)
            if step==0 and j==0:
                g=torch.autograd.grad(geo,pred,retain_graph=True)[0];assert g.abs().sum()>0 and torch.count_nonzero(g[:,0])==0
                gradient_checks['geometry_to_prediction']=float(g.norm());del g
            loss.backward(retain_graph=j+1<len(ctx))
            prefixes.append({'prefix':j+1,'latent_l1':float(latent.detach()),'geometry':float(geo.detach()),'state_rms':float(state.detach().square().mean().sqrt())})
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['clip']);assert torch.isfinite(norm) and norm>0
        if step==0:
            v=sum(float(o.weight.grad.abs().sum()) for o in model.o);assert v>0;gradient_checks['zero_projection_gradient']=v
        if step==1:
            for name,p in [('initial_tokens',model.initial_tokens),('cross_attention',model.update_attention.in_proj_weight),('residual_projection',model.residual_in.weight)]:
                v=float(p.grad.abs().sum());assert v>0;gradient_checks[name]=v
        assert all(p.grad is None for p in base.parameters()) and all(p.grad is None for p in probe.parameters())
        opt.step();scheduler.step()
        draws.append({'task':row['task'],'query':ids,'times':times,'context':ctx,'requested_k':requested,'actual_k':len(ctx),'query_successes':int(store.labels[ids].sum()),'prefixes':prefixes})
        l=np.mean([r['latent_l1'] for r in prefixes]);g=np.mean([r['geometry'] for r in prefixes]);running+=[l,g,l+cfg['geometry_weight']*g];nrun+=1
        if (step+1)%100==0 or step+1==steps:
            log={'step':step+1,'mean_latent_l1':running[0]/nrun,'mean_geometry':running[1]/nrun,'mean_loss':running[2]/nrun,'steps_in_average':nrun,'lr':scheduler.get_last_lr()[0],'seconds_this_attempt':time.time()-t0}
            logs.append(log);f.write_json(folder/'progress.json',log);print('TOKEN_TRAIN',seed,json.dumps(log),flush=True);running*=0;nrun=0
        if step+1 in checks:
            ck=save_checkpoint(folder,model,m,seed,step+1,prov);f.curves(root,folder,m,seed,step+1,ck,base,model,heads,rule,store,cross,smoke)
        if (step+1)%250==0 or step+1 in checks or step+1==steps:
            f.atomic_torch(resume,{'weights':f.weights(model),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'step':step+1,'steps':steps,'seed':seed,'provenance':prov,'draws':draws,'logs':logs,'gradient_checks':gradient_checks})
    assert fixed=={k:f.frozen_fingerprint(v) for k,v in [('base',base),('probe',probe)]};assert initial!=f.frozen_fingerprint(model)
    final=folder/'checkpoints'/f'step_{steps:06d}.pt';f.write_json(folder/'draws.json',draws)
    f.write_json(folder/'complete.json',{'complete':True,'steps':steps,'seed':seed,'provenance':prov,'checkpoint':str(final),'checkpoint_sha256':f.digest(final),'initial_model_sha256':initial,
        'frozen_modules_unchanged':True,'gradient_checks':gradient_checks,'actual_k':dict(Counter(r['actual_k'] for r in draws)),
        'scenes_seen':len({r['task'] for r in draws}),'unique_supports':len({i for r in draws for i in r['context']}),'logs':logs})
    if smoke:
        assert any(len(r['context'])>1 for r in draws)
        f.write_json(root/'SMOKE.json',{'passed':True,'provenance':prov,'steps':steps,'gradient_checks':gradient_checks,'scope':'Engineering only; recurrent gradients and fixed-query arrays; no effect gate'})
        print('TOKEN_SMOKE_PASSED',flush=True)


def evaluate(root,seed,within):
    m=f.read(root/'manifest.json');prov=f.provenance(root);assert m['code']==f.code_hashes()
    done=f.read(root/'models'/f'seed_{seed}'/'complete.json');assert done['complete'] and done['steps']==m['training']['steps'] and done['provenance']==prov
    ck=Path(done['checkpoint']);assert f.digest(ck)==done['checkpoint_sha256'];base,probe,rule,model,heads=models(m,seed,ck);prov={**prov,'checkpoint':f.digest(ck),'seed':seed}
    if within:
        wm=f.read(root/'within_inputs.json');assert wm['provenance']==f.provenance(root) and f.digest(wm['pack']+'_meta.npz')==wm['meta_sha256']
        specs=wm['specs'];store=f.Pack(wm['pack']);cross=None;prov['within_inputs']=f.digest(root/'within_inputs.json')
    else:
        specs=m['final_panels'];store=f.Store();cross=f.Pack(m['cross_pack']);assert f.digest(m['cross_pack']+'_meta.npz')==m['cross_meta_sha256']
    folder=root/('within_evaluation' if within else 'evaluation')/f'seed_{seed}';t0=time.time()
    for spec in specs:
        source=cross if spec['panel']=='cross_natural' else store;replay=f.PROBES/'evaluation/cross'/spec['task'].replace(':','_') if spec['panel']=='cross_natural' else None
        f.evaluate_panel(folder/spec['panel']/spec['task'].replace(':','_'),spec,source,base,model,heads,rule,prov,replay=replay)
        print('TOKEN_EVAL',seed,spec['panel'],spec['task'],round(time.time()-t0),flush=True)
    f.write_json(folder/'complete.json',{'complete':True,'provenance':prov,'specs':len(specs)})


@torch.no_grad()
def score_candidates(model,state,updates,z0,actions,initial_features,head,rule):
    values=[];memory,mask=model.state_memory(state,updates>0)
    for start in range(0,len(actions),32):
        ii=slice(start,start+32);z=torch.from_numpy(z0[ii].copy()).cuda().float();a=torch.from_numpy(actions[ii]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z,a,memory.expand(len(z),-1,-1),mask.expand(len(z),-1)).float()
        values.append(f.readout(head,pred,initial_features[ii],rule)['log_score'])
    result=np.concatenate(values);assert np.isfinite(result).all();return result


def log_area(first,budget):
    w=np.log(np.arange(2,budget+2))-np.log(np.arange(1,budget+1))
    return float(w[np.arange(1,budget+1)>=first].sum()/w.sum()) if first else 0.


@torch.no_grad()
def closed(root,seed,within,smoke=False):
    m=f.read(root/'manifest.json');assert m['code']==f.code_hashes();prov=f.provenance(root)
    done=f.read(root/('smoke' if smoke else 'models')/f'seed_{seed}'/'complete.json');assert done['provenance']==prov
    ck=Path(done['checkpoint']);assert f.digest(ck)==done['checkpoint_sha256'];base,probe,rule,model,heads=models(m,seed,ck);model.eval();head=heads[m['primary_head']]
    parameter_before=f.frozen_fingerprint(model)
    if within:
        wm=f.read(root/'within_inputs.json');assert wm['provenance']==prov and f.digest(wm['pack']+'_meta.npz')==wm['meta_sha256']
        source=f.Pack(wm['pack']);specs=f.choose_tasks(wm['specs'],m['closed_loop']['within_per_template'],'token_closed_within');prov['within_inputs']=f.digest(root/'within_inputs.json')
    else:
        source=f.Pack(m['cross_pack']);assert f.digest(m['cross_pack']+'_meta.npz')==m['cross_meta_sha256']
        specs=[r for r in m['final_panels'] if r['panel']=='cross_natural' and r['task'] in m['closed_loop']['cross_tasks']]
    if smoke:specs=[next(r for r in m['final_panels'] if r['panel']=='cross_natural' and r['context'] and source.labels[r['query']].any())]
    budget=3 if smoke else m['closed_loop']['budget'];prov={**prov,'checkpoint':f.digest(ck),'seed':seed,'budget':budget}
    folder=root/('closed_smoke' if smoke else 'closed_within' if within else 'closed_cross')/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    results=[];t0=time.time()
    for spec in specs:
        task=spec['task'];path=folder/task.replace(':','_');path.mkdir(exist_ok=True)
        if (path/'trace.json').exists():
            trace=f.read(path/'trace.json');assert trace['provenance']==prov;results.append(trace['result']);continue
        ids=spec['indices'];assert len(ids)==1024
        z0=np.array(source.lat[0][ids,0]);actions=source.actions[ids]
        # Only initial geometry is passed to candidate scoring. Future rows retrieved only on observation.
        initial_features=[source.feature(i)[:1].copy() for i in ids]
        state=model.initial_state().detach().clone();snapshots=[];states=[state.cpu().numpy().copy()];attempts=[];tried=np.zeros(len(ids),bool);updates=0;first=0
        score=score_candidates(model,state,0,z0,actions,initial_features,head,rule)
        reference=(f.WITHIN/'base_cache'/task.replace(':','_')/(m['primary_head']+'.npz')) if within else (f.PROBES/'evaluation/cross'/task.replace(':','_')/(m['primary_head']+'__base.npz'))
        with np.load(reference) as d:np.testing.assert_array_equal(score,d['log_score'])
        baseline_score=score.copy();snapshots.append(score.copy())
        for attempt in range(1,budget+1):
            local=int(np.argmax(np.where(tried,-np.inf,score)));idx=ids[local];assert not tried[local];tried[local]=True
            solved=bool(source.labels[idx])
            attempts.append({'attempt':attempt,'pack_index':idx,'candidate_local':local,'score_snapshot':updates,'score':float(score[local]),'solved':solved})
            if solved:first=attempt;break
            if attempt<budget:
                observed=torch.from_numpy(source.z([idx])).cuda().float();a=torch.from_numpy(source.actions[[idx]]).cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    residual=model.context_residual(observed[:,None],a[:,None])[:,0].float()
                    state=model.update_state(state,residual,a)
                assert torch.isfinite(state).all();updates+=1
                states.append(state.cpu().numpy().copy())
                score=score_candidates(model,state,updates,z0,actions,initial_features,head,rule);snapshots.append(score.copy())
        y=source.labels[ids];order=np.argsort(-baseline_score,kind='stable')[:budget]
        br=int(np.flatnonzero(y[order])[0])+1 if y[order].any() else 0
        result={'task':task,'first_success':first,'base_first_success':br,'attempts':len(attempts),'updates':updates,'log_area20' if budget==20 else 'log_area':log_area(first,budget),'base_log_area':log_area(br,budget)}
        f.save_npz(path/'snapshots.npz',{'indices':np.asarray(ids),'labels':y,'scores':np.stack(snapshots),'tokens':np.concatenate(states)})
        f.write_json(path/'trace.json',{'provenance':prov,'task':task,'attempts':attempts,'result':result,'base_reference':str(reference),'base_reference_sha256':f.digest(reference),
            'state_shape':list(state.shape),'residual_rule':m['residual'],'parameters_updated_online':False})
        results.append(result);print('TOKEN_CLOSED',seed,task,first,br,updates,round(time.time()-t0),flush=True)
    assert parameter_before==f.frozen_fingerprint(model)
    f.write_json(folder/'complete.json',{'complete':True,'provenance':prov,'tasks':[r['task'] for r in specs],'results':results,'weights_unchanged_online':True})


def summarize_closed(root,within,smoke=False):
    m=f.read(root/'manifest.json');folder=root/('closed_smoke' if smoke else 'closed_within' if within else 'closed_cross');seeds=[0] if smoke else m['seeds'];allresults={};checked=0
    if within:
        wm=f.read(root/'within_inputs.json');source=f.Pack(wm['pack']);expected=[r['task'] for r in f.choose_tasks(wm['specs'],m['closed_loop']['within_per_template'],'token_closed_within')]
    else:
        source=f.Pack(m['cross_pack']);expected=[r['task'] for r in m['final_panels'] if r['panel']=='cross_natural' and r['task'] in m['closed_loop']['cross_tasks']]
    for seed in seeds:
        done=f.read(folder/f'seed_{seed}'/'complete.json');assert done['complete'] and done['weights_unchanged_online']
        assert done['provenance']['manifest']==f.digest(root/'manifest.json') and done['provenance']['code']==f.code_hashes()
        md=f.read(root/('smoke' if smoke else 'models')/f'seed_{seed}'/'complete.json');assert done['provenance']['checkpoint']==md['checkpoint_sha256']==f.digest(md['checkpoint'])
        if not smoke:assert done['tasks']==expected
        results=[];budget=done['provenance']['budget']
        for task in done['tasks']:
            path=folder/f'seed_{seed}'/task.replace(':','_');trace=f.read(path/'trace.json');assert trace['provenance']==done['provenance'] and not trace['parameters_updated_online']
            with np.load(path/'snapshots.npz') as d:ids=d['indices'];y=d['labels'];scores=d['scores'];tokens=d['tokens']
            np.testing.assert_array_equal(ids,source.by[task]);np.testing.assert_array_equal(y,source.labels[ids]);assert np.isfinite(scores).all() and np.isfinite(tokens).all()
            assert f.digest(trace['base_reference'])==trace['base_reference_sha256']
            with np.load(trace['base_reference']) as b:np.testing.assert_array_equal(scores[0],b['log_score'])
            tried=np.zeros(len(ids),bool);update=0;first=0
            for a,t in enumerate(trace['attempts'],1):
                assert t['attempt']==a and t['score_snapshot']==update
                j=int(np.argmax(np.where(tried,-np.inf,scores[update])))
                assert j==t['candidate_local'] and int(ids[j])==t['pack_index'] and scores[update,j]==t['score'] and bool(y[j])==t['solved']
                assert not tried[j];tried[j]=True;checked+=1
                if y[j]:first=a;assert a==len(trace['attempts']);break
                if a<budget:update+=1
            assert len(trace['attempts'])==(first or budget) and len(scores)==len(tokens)==update+1
            if update:assert any(not np.array_equal(tokens[i],tokens[i-1]) for i in range(1,len(tokens)))
            order=np.argsort(-scores[0],kind='stable')[:budget];br=int(np.flatnonzero(y[order])[0])+1 if y[order].any() else 0
            r=trace['result'];assert r['first_success']==first and r['base_first_success']==br and r['updates']==update
            assert abs(r['base_log_area']-log_area(br,budget))<1e-12
            key='log_area20' if budget==20 else 'log_area';assert abs(r[key]-log_area(first,budget))<1e-12
            assert r==next(x for x in done['results'] if x['task']==task);results.append(r)
        metrics={}
        for kind,field in [('memory','first_success'),('base','base_first_success')]:
            metrics[kind]={f'success_at_{k}':float(np.mean([0<r[field]<=k for r in results])) for k in [1,5,10,20] if k<=budget}
            metrics[kind]['log_area_budget_normalized']=float(np.mean([log_area(r[field],budget) for r in results]))
        allresults[str(seed)]={'metrics':metrics,'per_task':results}
    out={'complete':True,'trace_audit_passed':True,'provenance':f.provenance(root),'attempts_replayed':checked,'budget':budget,'per_seed':allresults,'scope':'Cached simulator closed-loop replay; tokens update after every failure with remaining budget; no online weight changes; 20-attempt area is not canonical AUCCESS'}
    f.write_json(root/('CLOSED_SMOKE_AUDIT.json' if smoke else 'CLOSED_WITHIN_SUMMARY.json' if within else 'CLOSED_CROSS_SUMMARY.json'),out)
    print('TOKEN_CLOSED_AUDIT_PASSED',checked,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','train','eval','summary','prepare_within','closed','closed_summary']);p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--within',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare(a.root)
    elif a.stage in ['smoke','train']:train(a.root,a.seed,a.stage=='smoke')
    elif a.stage=='eval':evaluate(a.root,a.seed,a.within)
    elif a.stage=='summary':f.summarize(a.root,a.within)
    elif a.stage=='prepare_within':f.prepare_within(a.root)
    elif a.stage=='closed':closed(a.root,a.seed,a.within,a.smoke)
    elif a.stage=='closed_summary':summarize_closed(a.root,a.within,a.smoke)
if __name__=='__main__':main()
