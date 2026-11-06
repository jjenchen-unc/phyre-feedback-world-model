"""Full-budget active-prefix predictor, then causal scorer-policy updater training."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import query_token_staged as old
from prefix_policy_predictor import PrefixPredictor, policy_rollout, replay_state, choose_action
f=old.f
STAGED=f.PROJECT/'results/query_token_staged_20261005'
COMMON=f.PROJECT/'results/prefix_policy_20261008'


def reader(name):return name.startswith('base.') or name=='initial_tokens'
def fingerprint(model, which):
    h=hashlib.sha256()
    for name,p in sorted(model.state_dict().items()):
        if reader(name)==(which=='reader'):
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def checked(root):
    m=f.read(root/'manifest.json')
    assert m['code']==f.code_hashes()
    assert f.read(root/'DATA_AUDIT.json')['manifest']==f.digest(root/'manifest.json')
    for path,sha in m['verified_files'].items():assert f.digest(path)==sha,path
    return m


def prepare(root):
    assert not (root/'manifest.json').exists()
    parent=f.read(STAGED/'manifest.json');prep=f.read(COMMON/'PREPARATION.json')
    assert f.read(COMMON/'SMOKE.json')['passed']
    assert f.read(COMMON/'SMOKE.json')['preparation_sha256']==f.digest(COMMON/'PREPARATION.json')
    for name,sha in prep['code'].items():assert f.digest(COMMON/'code'/name)==sha
    assert f.digest(Path(__file__).parent/'prefix_policy_predictor.py')==prep['code']['prefix_policy_predictor.py']
    assert f.read(STAGED/'DATA_AUDIT.json')['provenance']['manifest']==f.digest(STAGED/'manifest.json')
    m={k:parent[k] for k in ['base','base_sha256','probe','probe_sha256','heads','primary_head',
        'cross_pack','cross_meta_sha256','source_hashes','store_files_sha256','stage1_pack','stage1_meta_sha256',
        'stage1_dev_pack','stage1_dev_meta_sha256']}
    within=f.read(STAGED/'within_inputs.json')
    m.update(seeds=[0,1,2],code=f.code_hashes(),rows=prep['rows'],data_counts=prep['data_counts'],
        architecture={'n_tokens':32,'mem_steps':16,'mem_grid':10,'checkpoint_blocks':True},
        stage1={'epochs':8,'batch':8,'n_train':52276,'updates':52272,'presentations':418176,
            'initialization':'original pretrained predictor plus fresh active self-attention prefix Q0',
            'trainable':'backbone and Q0 only'},
        stage2={'episodes':20000,'scene_batch':2,'scheduled_slots':10000,'queries':8,'prefix_caps':[1,2,4,8],
            'branches':['residual','zero'],'trainable':'updater only; frozen backbone and Q0',
            'supervision':'query loss after the last observed failed support up to a uniformly drawn cap; stop at success',
            'zero_support':'retain diagnostic query loss, no updater gradient; skip optimizer if both scenes have no support',
            'weighting':'each scene loss divided by two, including no-support scenes; no failure resampling',
            'scheduler':'OneCycleLR over scheduled slots including skipped optimizer slots',
            'policy':'rescore mixed candidates with current tokens before every selected trial; detached argmax',
            'residual':'chosen truth minus current token-conditioned prediction; full BPTT replay at unchanged weights',
            'replay_chunk':32},
        training={'lr':3e-4,'weight_decay':.05,'clip':1.,'geometry_weight':.1,'scheduler':'OneCycleLR 3%warmup'},
        evaluation={'primary':'S@9','budgets':[1,3,5,9,20],'scenes_per_split':100,'choice_chunk':32,
            'cross_specs':[s for s in parent['final_panels'] if s['panel']=='cross_natural'],
            'within_pack':within['pack'],'within_meta_sha256':within['meta_sha256'],'within_specs':within['specs'],
            'selection':'final fixed checkpoints; no selection on evaluation results'},
        limits=['Stage1 adds full base-sized training on top of pretraining; objective L1+0.1geometry differs from historical L1',
            'Stage2 has 1200 scenes; stage1 has1600 including updater calibration templates',
            'Training support-half pool is smaller than deployment1024 pool; actual scorer policy within that pool',
            'No-support episodes reduce optimizer updates; report exposure and realized updates separately by branch',
            'Zero-residual policy may select different supports; contrast includes policy distribution changes',
            'Full BPTT through residual updater and predictor; no gradient through discrete action selection',
            'Existing cross/within dev panels reused; within generated same-template, not official test',
            'S@20 stress exceeds maximum8 trained updates; three seeds do not establish broad significance'])
    verified={str(STAGED/'manifest.json'):f.digest(STAGED/'manifest.json'),str(COMMON/'PREPARATION.json'):f.digest(COMMON/'PREPARATION.json')}
    verified.update(m['source_hashes'])
    verified.update({m['base']:m['base_sha256'],m['probe']:m['probe_sha256'],
        m['stage1_pack']+'_meta.npz':m['stage1_meta_sha256'],m['stage1_dev_pack']+'_meta.npz':m['stage1_dev_meta_sha256'],
        m['cross_pack']+'_meta.npz':m['cross_meta_sha256'],within['pack']+'_meta.npz':within['meta_sha256']})
    for path,sha in verified.items():assert f.digest(path)==sha
    source=f.Pack(m['stage1_pack']);store=f.Store(include_features=False)
    assert len(source.actions)==52276 and len(source.by)==1600
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest()==m['store_files_sha256']
    inv=f.read(f.PROJECT/'results/support_utility_20261004/inventory.json')
    assert f.digest(f.PROJECT/'results/support_utility_20261004/inventory.json')==parent['inventory_sha256']
    inv={r['task']:r for r in inv['rows']}
    for r in m['rows']:
        a,b=r['candidates'],r['query'];groups=inv[r['task']]['groups']
        assert a==inv[r['task']]['support'] and not set(a)&set(b)
        assert not {groups[str(i)]['group'] for i in a}&{groups[str(i)]['group'] for i in b}
        assert all(str(store.tasks[i])==r['task'] for i in a+b)
        assert not {tuple(store.actions[i]) for i in a}&{tuple(store.actions[i]) for i in b}
    cross=m['evaluation']['cross_specs'];withinrows=m['evaluation']['within_specs']
    assert len(cross)==len(withinrows)==100
    assert not {s['task'][:5] for s in cross}&{t[:5] for t in source.by}
    assert not {s['task'] for s in withinrows}&set(source.by)
    draws={}
    for seed in m['seeds']:
        vv=[old.epoch_indices(52276,seed,ep) for ep in range(8)]
        assert all(v.shape==(6534,8) and len(np.unique(v))==52272 for v in vv)
        draws[str(seed)]=hashlib.sha256(b''.join(v.tobytes() for v in vv)).hexdigest()
    assert draws==f.read(STAGED/'DATA_AUDIT.json')['stage1_epoch_draw_sha256']
    m['verified_files']=verified
    f.write_json(root/'manifest.json',m)
    f.write_json(root/'DATA_AUDIT.json',{'passed':True,'manifest':f.digest(root/'manifest.json'),
        'stage1_epoch_draw_sha256':draws,'same_stage1_draws_as_previous_staged':True,
        'mixed_candidates_group_disjoint':True,'cross_template_disjoint':True,'within_task_disjoint':True})


def model_for(root,stage,seed,checkpoint=None,parent_checkpoint=None):
    m=checked(root);ref,probe,rule,pc,_=f.load_models()
    ref.eval().requires_grad_(False);probe.eval().requires_grad_(False)
    torch.manual_seed(seed+7919)
    model=PrefixPredictor(copy.deepcopy(ref),pc['Din'],stage='initial' if stage=='q0' else 'updater',
        zero_residual=stage=='zero',**m['architecture']).cuda()
    if stage!='q0':
        if parent_checkpoint is None:
            done=f.read(root/'models'/f'q0_s{seed}'/'complete.json')
            assert done['provenance']==f.provenance(root) and done['slots']==52272
            parent_checkpoint=Path(done['checkpoint']);assert f.digest(parent_checkpoint)==done['checkpoint_sha256']
        ck=torch.load(parent_checkpoint,map_location='cpu',weights_only=False)
        assert ck['stage']=='q0' and ck['seed']==seed and ck['provenance']==f.provenance(root)
        state=model.state_dict();state.update({k:v for k,v in ck['state_dict'].items() if reader(k)})
        model.load_state_dict(state);model.reset_updater_identity()
    if checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['stage']==stage and ck['seed']==seed and ck['provenance']==f.provenance(root)
        model.load_state_dict(ck['state_dict'])
    heads,hh=f.load_heads(m,f.PROBES,probe)
    assert hh[m['primary_head']]==m['heads'][m['primary_head']]
    return ref,probe,rule,model,heads[m['primary_head']],parent_checkpoint


def score_function(head,rule):
    def score(pred,features):
        with torch.autocast('cuda',enabled=False):value=f.readout(head,pred.float(),features,rule)['log_score']
        return torch.from_numpy(value).to(pred.device)
    return score


def initial_data(source,ids):
    # Only first-frame latents and features enter the candidate selector.
    refs=[source.refs[int(i)] for i in ids] if hasattr(source,'refs') else [(0,int(i)) for i in ids]
    z0=torch.from_numpy(np.stack([source.lat[p][j,0] for p,j in refs])).cuda().float()
    actions=torch.from_numpy(source.actions[ids]).cuda()
    return z0,actions,[source.feature(i)[:1].copy() for i in ids]


def episode(rows,seed,index):
    rng=np.random.default_rng([20261008,seed,index,0]);row=rows[int(rng.integers(len(rows)))]
    slots=rng.choice(len(row['query']),min(8,len(row['query'])),replace=False).tolist()
    cap=int(rng.choice([1,2,4,8]))
    return row,[row['query'][j] for j in slots],[row['times'][j] for j in slots],cap


def updater_loss(model,probe,head,rule,source,ep,verify=False):
    row,qi,times,cap=ep;ids=row['candidates'];z0,actions,features=initial_data(source,ids)
    def truth(j):return torch.from_numpy(source.z([ids[j]])).cuda().float()
    def observe(j):return bool(source.labels[ids[j]]),truth(j)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        trace,supports=policy_rollout(model,z0,actions,features,score_function(head,rule),observe,budget=cap+1)
    assert len({t['local'] for t in trace})==len(trace) and not any(t['solved'] for t in trace[:-1])
    assert all(not source.labels[ids[j]] for j in supports)
    with torch.autocast('cuda',dtype=torch.bfloat16):state=replay_state(model,z0,actions,supports,truth)
    if verify:assert torch.equal(state.detach().cpu(),trace[-1]['state'])
    z=torch.from_numpy(source.z(qi)).cuda().float();qa=torch.from_numpy(source.actions[qi]).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z[:,0],qa,state.expand(len(qi),-1,-1)).float()
    l=(pred[:,1:]-z[:,1:]).abs().mean();g=f.geometry_loss(probe,pred,[source.feature(i) for i in qi],times)
    loss=l+.1*g
    assert loss.requires_grad==bool(supports)
    event={'task':row['task'],'query':qi,'cap':cap,'supports':[ids[j] for j in supports],
        'selected':[ids[t['local']] for t in trace],'solved':[t['solved'] for t in trace],
        'used_updates':len(supports),'loss':float(loss.detach()),'l1':float(l.detach()),'geometry':float(g.detach())}
    return loss,event


def save(root,folder,model,stage,seed,slot):
    path=folder/'checkpoints'/f'step_{slot:06d}.pt'
    f.atomic_torch(path,{'stage':stage,'seed':seed,'slots':slot,'provenance':f.provenance(root),
        'state_dict':{n:p.detach().cpu() for n,p in model.state_dict().items()}})
    return path


def train(root,stage,seed,smoke=False,parent_checkpoint=None):
    m=checked(root);prov=f.provenance(root)
    if not smoke:assert f.read(root/'SMOKE.json')['passed'] and f.read(root/'SMOKE.json')['provenance']==prov
    folder=root/('smoke' if smoke else 'models')/f'{stage}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        d=f.read(folder/'complete.json');assert d['provenance']==prov and f.digest(d['checkpoint'])==d['checkpoint_sha256'];return d
    ref,probe,rule,model,head,parent_checkpoint=model_for(root,stage,seed,parent_checkpoint=parent_checkpoint)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert names and all(reader(n)==(stage=='q0') for n in names)
    source=f.Pack(m['stage1_pack']) if stage=='q0' else f.Store()
    rows=[r for r in m['rows'] if r['split']=='train']
    total=(40 if stage=='q0' else 32) if smoke else (52272 if stage=='q0' else 10000)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.05)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-4,total_steps=total,pct_start=.03)
    frozen_part='updater' if stage=='q0' else 'reader';frozen_before=fingerprint(model,frozen_part)
    initial_model=f.frozen_fingerprint(model);ref_hash=f.frozen_fingerprint(ref);probe_hash=f.frozen_fingerprint(probe)
    ids=[0,1];zz=torch.from_numpy(source.z(ids)).cuda().float();aa=torch.from_numpy(source.actions[ids]).cuda()
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):initial_prediction=model(zz[:,0],aa).clone()
    start=0;updates=0;logs=[];draws=[];positive=0;empty=0;actual_supports=0;resume=folder/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['provenance']==prov and ck['total']==total and ck['stage']==stage and ck['seed']==seed
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['slots'];updates=ck['optimizer_steps'];logs=ck['logs'];draws=ck['draws']
        positive=ck['positive'];empty=ck['empty'];actual_supports=ck['actual_supports'];initial_model=ck['initial_model_sha256']
    dev=f.Pack(m['stage1_dev_pack']) if stage=='q0' else None
    if stage=='q0' and start==0:f.write_json(folder/'dev_initial.json',old.dev_l1(model,dev,16 if smoke else None))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.time();running=[];epoch_order=None
    for slot in range(start,total):
        model.train();ref.eval();probe.eval();opt.zero_grad(set_to_none=True)
        has_gradient=False
        if stage=='q0':
            epoch,offset=divmod(slot,6534)
            if epoch_order is None or offset==0:epoch_order=old.epoch_indices(52276,seed,epoch)
            ids=epoch_order[offset].tolist();loss,l,g=old.q0_loss(model,probe,source,ids)
            assert torch.isfinite(loss);loss.backward();has_gradient=True
            draws.append(ids);running.append([float(l.detach()),float(g.detach()),float(loss.detach())])
            if slot==0:assert model.initial_tokens.grad is not None and model.initial_tokens.grad.abs().sum()>0
        else:
            # Two scene graphs are backpropagated sequentially at unchanged weights.
            for j in range(2):
                ep=episode(rows,seed,2*slot+j)
                loss,event=updater_loss(model,probe,head,rule,source,ep,verify=smoke or slot<3)
                assert torch.isfinite(loss)
                if loss.requires_grad:(loss/2).backward();has_gradient=True;positive+=1
                else:empty+=1
                actual_supports+=event['used_updates'];draws.append(event)
                running.append([event['l1'],event['geometry'],event['loss']])
                del loss
        if has_gradient:
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(norm) and norm>0
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            opt.step();updates+=1
        sched.step();done=slot+1
        if done%100==0 or done==total or (smoke and done%8==0):
            torch.cuda.synchronize()
            row={'slots':done,'optimizer_steps':updates,'losses':np.mean(running,axis=0).tolist(),
                'seconds_this_attempt':time.time()-t0,'start_slot':start,'positive_episodes':positive,
                'zero_support_episodes':empty,'observed_failure_updates':actual_supports,
                'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,'lr':sched.get_last_lr()[0]}
            logs.append(row);running=[];f.write_json(folder/'progress.json',row);print('TRAIN',stage,seed,json.dumps(row),flush=True)
        epoch_end=stage=='q0' and done%6534==0
        milestone=done==total or epoch_end or (stage!='q0' and done in [500,2000,5000])
        if milestone:
            save(root,folder,model,stage,seed,done)
            if stage=='q0':f.write_json(folder/f'dev_{done:06d}.json',old.dev_l1(model,dev,16 if smoke else None))
        if done%500==0 or milestone:
            assert fingerprint(model,frozen_part)==frozen_before
            f.atomic_torch(resume,{'provenance':prov,'stage':stage,'seed':seed,'total':total,'slots':done,
                'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                'optimizer_steps':updates,'logs':logs,'draws':draws,'positive':positive,'empty':empty,
                'actual_supports':actual_supports,'initial_model_sha256':initial_model})
    assert f.frozen_fingerprint(ref)==ref_hash and f.frozen_fingerprint(probe)==probe_hash
    assert fingerprint(model,frozen_part)==frozen_before
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=model(zz[:,0],aa)
    if stage!='q0':assert torch.equal(initial_prediction,after)
    else:assert not torch.equal(initial_prediction,after)
    assert updates>0
    path=save(root,folder,model,stage,seed,total)
    result={'complete':True,'stage':stage,'seed':seed,'slots':total,'optimizer_steps':updates,
        'provenance':prov,'checkpoint':str(path),'checkpoint_sha256':f.digest(path),'logs':logs,
        'frozen_sha256':frozen_before,'initial_model_sha256':initial_model,'trainable_names':names,
        'positive_episodes':positive,'zero_support_episodes':empty,'observed_failure_updates':actual_supports,
        'draw_sha256':hashlib.sha256(json.dumps(draws,sort_keys=True).encode()).hexdigest(),
        'parent_checkpoint_sha256':f.digest(parent_checkpoint) if parent_checkpoint else None,
        'checks':{'frozen_unchanged':True,'q0_unchanged_stage2':stage!='q0','active_q0_trained':stage=='q0'}}
    f.write_json(folder/'draws.json',draws);f.write_json(folder/'complete.json',result);return result


@torch.no_grad()
def static_scores(model,z0,actions,features,score):
    vals=[]
    for start in range(0,len(actions),32):
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z0[start:start+32],actions[start:start+32])
        vals.append(score(pred,features[start:start+32]).cpu().numpy())
    return np.concatenate(vals)


def first_rank(scores,labels):
    hits=np.flatnonzero(labels[np.argsort(-scores,kind='stable')]);return int(hits[0])+1 if len(hits) else None


@torch.no_grad()
def eval_case(folder,spec,source,model,ref,head,rule,prov,stage,q0file=None,budget=20):
    if (folder/'result.json').exists():
        d=f.read(folder/'result.json');assert d['provenance']==prov;audit_case(folder);return d
    folder.mkdir(parents=True,exist_ok=True);ids=spec['indices']
    assert all(str(source.tasks[i])==spec['task'] for i in ids)
    z0,actions,features=initial_data(source,ids);score=score_function(head,rule)
    model.eval();ref.eval();labels=source.labels[ids]
    if stage=='q0':
        original=static_scores(ref,z0,actions,features,score);qs=static_scores(model,z0,actions,features,score)
        raw={'indices':np.asarray(ids),'labels':labels,'original':original,'q0':qs}
        d={'task':spec['task'],'stage':stage,'provenance':prov,'first_success':first_rank(qs,labels),
            'original_first_success':first_rank(original,labels)}
    else:
        def observe(j):return bool(source.labels[ids[j]]),torch.from_numpy(source.z([ids[j]])).cuda().float()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            trace,supports=policy_rollout(model,z0,actions,features,score,observe,budget=budget)
        snapshots=np.stack([t['scores'].numpy() for t in trace]);sel=np.array([t['local'] for t in trace])
        with np.load(q0file) as base:
            np.testing.assert_array_equal(base['indices'],ids);np.testing.assert_array_equal(base['labels'],labels)
            np.testing.assert_array_equal(base['q0'],snapshots[0])
        raw={'indices':np.asarray(ids),'labels':labels,'scores':snapshots,'selected':sel}
        d={'task':spec['task'],'stage':stage,'provenance':prov,'first_success':len(trace) if trace[-1]['solved'] else None,
            'attempts':len(trace),'updates':len(supports),'budget':budget}
    f.save_npz(folder/'scores.npz',raw);f.write_json(folder/'result.json',d);audit_case(folder);return d


def audit_case(folder):
    d=f.read(folder/'result.json')
    with np.load(folder/'scores.npz') as raw:
        assert len(np.unique(raw['indices']))==len(raw['indices'])
        if d['stage']=='q0':
            assert first_rank(raw['q0'],raw['labels'])==d['first_success']
            assert first_rank(raw['original'],raw['labels'])==d['original_first_success']
        else:
            tried=set();hit=None
            for attempt,(scores,j) in enumerate(zip(raw['scores'],raw['selected']),1):
                assert np.isfinite(scores).all();scores=scores.copy();scores[list(tried)]=-np.inf
                assert int(np.argmax(scores))==int(j) and int(j) not in tried;tried.add(int(j))
                if raw['labels'][j]:assert attempt==len(raw['selected']);hit=attempt
            assert hit==d['first_success'] and len(tried)==d['attempts']
            assert d['updates']==d['attempts']-1 and d['attempts']<=d['budget']
    return d


def evaluate(root,stage,seed,split):
    m=checked(root);done=f.read(root/'models'/f'{stage}_s{seed}'/'complete.json')
    assert done['provenance']==f.provenance(root) and done['slots']==(52272 if stage=='q0' else 10000)
    ck=Path(done['checkpoint']);assert f.digest(ck)==done['checkpoint_sha256']
    ref,probe,rule,model,head,_=model_for(root,stage,seed,ck)
    ev=m['evaluation'];source=f.Pack(m['cross_pack'] if split=='cross' else ev['within_pack']);specs=ev[split+'_specs']
    folder=root/'evaluation'/split/f'{stage}_s{seed}';prov={**f.provenance(root),'checkpoint':f.digest(ck),'stage':stage,'seed':seed,'split':split}
    results=[];t0=time.time()
    for spec in specs:
        name=spec['task'].replace(':','_');q0file=root/'evaluation'/split/f'q0_s{seed}'/name/'scores.npz'
        d=eval_case(folder/name,spec,source,model,ref,head,rule,prov,stage,q0file)
        results.append(d);f.write_json(folder/'progress.json',{'scenes':len(results),'total':100,'seconds':time.time()-t0})
        print('EVAL',stage,seed,split,len(results),spec['task'],flush=True)
    assert len(results)==100
    f.write_json(folder/'complete.json',{'complete':True,'provenance':prov,'n':100,
        'counts':{f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in results) for k in ev['budgets']}})


def stress_replay(root,parent,checkpoint):
    ref,probe,rule,model,head,_=model_for(root,'residual',0,checkpoint,parent)
    source=f.Store();rows=[r for r in checked(root)['rows'] if r['split']=='train'
        and len(r['candidates'])>=9 and not source.labels[r['candidates']].any()]
    assert rows,'Need a natural all-failure scene for engineering maximum-depth stress'
    row=max(rows,key=lambda r:len(r['candidates']))
    ep=(row,row['query'][:8],row['times'][:8],8)
    torch.cuda.reset_peak_memory_stats();model.zero_grad(set_to_none=True)
    loss,event=updater_loss(model,probe,head,rule,source,ep,verify=True)
    assert event['used_updates']==8;loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    assert torch.isfinite(norm) and norm>0
    return {'passed':True,'engineering_only_all_failure_scene':row['task'],
        'candidate_count':len(row['candidates']),'depth':8,'queries':len(ep[1]),
        'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,'exact_replay':True}


def smoke(root):
    q0=train(root,'q0',0,True);parent=Path(q0['checkpoint'])
    branches=[train(root,s,0,True,parent) for s in ['residual','zero']]
    assert branches[0]['frozen_sha256']==branches[1]['frozen_sha256']
    assert branches[0]['initial_model_sha256']==branches[1]['initial_model_sha256']
    stress=stress_replay(root,parent,Path(branches[0]['checkpoint']))
    m=checked(root);source=f.Pack(m['cross_pack']);spec=next(s for s in m['evaluation']['cross_specs'] if s['task']=='00007:047')
    for stage,result in zip(['q0','residual','zero'],[q0]+branches):
        ref,probe,rule,model,head,_=model_for(root,stage,0,Path(result['checkpoint']),parent if stage!='q0' else None)
        folder=root/'smoke'/'evaluation'/stage
        eval_case(folder,spec,source,model,ref,head,rule,{'smoke':True,**f.provenance(root)},stage,
            root/'smoke'/'evaluation'/'q0'/'scores.npz',budget=9)
    f.write_json(root/'SMOKE.json',{'passed':True,'provenance':f.provenance(root),
        'scope':'40 Q0 updates;32 two-scene updater slots per branch; GPU exact policy replay; paired fresh init; causal full1024 evaluation',
        'max_depth_stress':stress,'stages':{r['stage']:r for r in [q0]+branches}})
    print('PREFIX_STAGED_SMOKE_PASS',flush=True)


def summarize(root):
    m=checked(root);out={}
    for seed in m['seeds']:
        q0=f.read(root/'models'/f'q0_s{seed}'/'complete.json')
        branches=[f.read(root/'models'/f'{s}_s{seed}'/'complete.json') for s in ['residual','zero']]
        assert q0['slots']==q0['optimizer_steps']==52272
        assert branches[0]['initial_model_sha256']==branches[1]['initial_model_sha256']
        assert branches[0]['frozen_sha256']==branches[1]['frozen_sha256']
        for r in branches:assert r['parent_checkpoint_sha256']==q0['checkpoint_sha256'] and r['slots']==10000 and r['positive_episodes']+r['zero_support_episodes']==20000
    for split in ['cross','within']:
        out[split]={}
        for seed in m['seeds']:
            for stage in ['q0','residual','zero']:
                folder=root/'evaluation'/split/f'{stage}_s{seed}';done=f.read(folder/'complete.json')
                assert done['n']==100 and done['provenance']['manifest']==f.digest(root/'manifest.json')
                rows=[audit_case(folder/s['task'].replace(':','_')) for s in m['evaluation'][split+'_specs']]
                counts={f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in rows) for k in m['evaluation']['budgets']}
                assert counts==done['counts']
                out[split][f'{stage}_s{seed}']={'counts':counts,'first_success':{r['task']:r['first_success'] for r in rows}}
                if stage=='q0':out[split][f'original_s{seed}']={'counts':{f's{k}':sum(r['original_first_success'] is not None and r['original_first_success']<=k for r in rows) for k in m['evaluation']['budgets']}}
            assert len({out[split][f'{s}_s{seed}']['counts']['s1'] for s in ['q0','residual','zero']})==1
    f.write_json(root/'SUMMARY.json',{'complete':True,'raw_audit_passed':True,'provenance':f.provenance(root),'splits':out,'limits':m['limits']})


def main():
    p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',default='q0',choices=['q0','residual','zero']);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--split',choices=['cross','within'],default='cross');a=p.parse_args();torch.set_num_threads(4)
    if a.command=='prepare':prepare(a.root)
    elif a.command=='smoke':smoke(a.root)
    elif a.command=='train':train(a.root,a.stage,a.seed)
    elif a.command=='eval':evaluate(a.root,a.stage,a.seed,a.split)
    elif a.command=='summary':summarize(a.root)
    else:raise ValueError(a.command)
if __name__=='__main__':main()
