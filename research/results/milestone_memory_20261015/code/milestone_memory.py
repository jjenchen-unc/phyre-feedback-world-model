"""Matched-Q0 milestone: legacy terminal-loss updater and Q0 predictor TTA.

Run frozen copies under results/milestone_memory_20261015/code. No old outputs
are changed; only immutable observation caches may be extended under file locks.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import validate_multiturn_runtime as v
from fast_updater import updater_loss
from fast_inference import CandidateRanker
from multiturn_runtime import PredictorAdapter

b,m,p=v.b,v.m,v.p
ROOT=v.ROOT/'results/milestone_memory_20261015'
BUDGET=9
TOTAL=10000


def read(path): return json.loads(Path(path).read_text())
def write(path,value): m.write(Path(path),value)
def sha(path): return m.sha(Path(path))
def ident(model): return v.fingerprint(model.state_dict().items())
def reader_hash(model):
    return v.fingerprint((n,x) for n,x in model.named_parameters()
                         if n=='initial_tokens' or n.startswith('base.'))


def provenance():
    return {'plan_sha256':sha(ROOT/'PLAN.json'), 'code_sha256':{
        str(x):sha(x) for x in sorted((ROOT/'code').glob('*.py'))}}


def plan_checked():
    plan=read(ROOT/'PLAN.json')
    for path,digest in plan['dependencies'].items():
        assert sha(path)==digest, ('Dependency changed',path)
    return plan


def prepare():
    assert not (ROOT/'PLAN.json').exists()
    previous=read(b.ROOT/'PLAN.json');panels={};dependencies={}
    for panel in ['cross','within']:
        weight=0. if panel=='cross' else .03
        initial={}
        for arch in ['base','q0']:
            folder=b.ROOT/'models'/panel/b.spec_name(arch,weight,0)
            complete=read(folder/'complete.json');path=Path(complete['checkpoint'])
            assert complete['complete'] and sha(path)==complete['checkpoint_sha256']
            initial[arch]={'checkpoint':str(path),'sha256':sha(path),
                'evaluation':str(b.ROOT/'evaluation'/panel/b.spec_name(arch,weight,0))}
        manifest=m.W/'manifest.json' if panel=='within' else m.OLD/'manifest.json'
        src=read(manifest);rows=[r for r in src['rows'] if r['split']=='train']
        tasks=previous['panels'][panel]['tasks'];assert len(tasks)==100 and len(rows)==1200
        assert not set(tasks)&{r['task'] for r in rows}
        if panel=='cross':assert not {t[:5] for t in tasks}&{r['task'][:5] for r in rows}
        for row in rows:
            assert not set(row['candidates'])&set(row['query'])
        panels[panel]={'initial_weight':weight,'fold':1 if panel=='cross' else 0,
            'initial':initial,'source_manifest':str(manifest),'source_manifest_sha256':sha(manifest),
            'store_files_sha256':src['store_files_sha256'],'tasks':tasks,
            'tta':{'lr':.001,'steps':1 if panel=='cross' else 3,'replay':5,
                   'selection':'Reuse historical protocol setting; no search on new Q0'},
            'source_counts':src.get('data_counts',{}),'cases':{}}
        for task in tasks:
            d,status,path=m.input_case(panel,task)
            cases={'shortlist':str(path),'shortlist_sha256':sha(path),
                   'labels_sha256':sha(path.with_name(path.stem+'_labels.npy'))}
            for arch in ['base','q0']:
                old=Path(initial[arch]['evaluation'])/task.replace(':','_')
                with np.load(old/'scores.npz') as z:
                    np.testing.assert_array_equal(z['indices'],d['action_idx'])
                    np.testing.assert_array_equal(z['labels'],status>0)
                    assert p.first_rank(z['scores'],status>0)==read(old/'result.json')['first_success']
                cases[arch+'_scores_sha256']=sha(old/'scores.npz')
            panels[panel]['cases'][task]=cases
        for path in [manifest,m.PROBES[panel],m.HEAD if panel=='cross' else m.PROBES[panel],m.PCA]:
            dependencies[str(path)]=sha(path)
    for path in list((m.R/'code').glob('*.py'))+[Path(m.__file__)]:
        dependencies[str(path)]=sha(path)
    plan={'date':'2026-10-15','scope':'Milestone, new Q0 with unchanged legacy updater recipe',
        'panels':panels,'seed':0,'methods':['base','q0','residual','zero','tta'],
        'training':{'slots':TOTAL,'episodes_per_slot':2,'queries':'min(8, available query actions)',
            'support_caps':[1,2,4,8],'loss':'Final updated-state query future latent L1 + 0.1 geometry',
            'geometry_weight':.1,'lr':3e-4,'weight_decay':.05,'clip':1.,
            'scheduler':'OneCycle pct_start=.03 over 10000 slots including no-gradient slots',
            'trainable':'fresh updater only, frozen predictor and initial tokens',
            'residual':'observed minus current token-conditioned prediction, full BPTT',
            'sampling':'legacy grouped query/support partition and per-episode RNG; stop at success',
            'checkpoint_blocks':True,'numerics':'Historical default CUDA; strict mode only for equivalence preflight',
            'checkpoint_every':500,'eta_every':1000},
        'evaluation':{'tasks_per_panel':100,'candidates':1024,'budget':BUDGET,
            'budgets':[1,3,5,9],'primary':'S@9','batch':32,
            'pool':'Existing corrected base-shortlist, identical to matched initial study',
            'initial_score_guard':'All 1024 scores equal fixed Q0 before every dynamic episode',
            'diagnostics':'Paired gained/lost solves and first-failure-update ranking on common untried actions'},
        'gpu_uuids':[], 'gpu_limit':6,'dependencies':dependencies}
    # The user specifically selected physical GPU 0..5 on oprime.
    import subprocess
    entries=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    plan['gpu_uuids']=[line.split(',')[1].strip() for line in entries.splitlines() if int(line.split(',')[0])<6]
    assert len(plan['gpu_uuids'])==6
    write(ROOT/'PLAN.json',plan);print('PREPARED',ROOT,flush=True)


def source_for(panel,plan):
    source,rows=v.dataset(panel)
    assert hashlib.sha256('\n'.join(source.files).encode()).hexdigest()==plan['panels'][panel]['store_files_sha256']
    for row in rows:
        assert all(str(source.tasks[i])==row['task'] for i in row['candidates']+row['query'])
        assert not set(row['candidates'])&set(row['query'])
        assert not {tuple(source.actions[i]) for i in row['candidates']} & {tuple(source.actions[i]) for i in row['query']}
    return source,rows


def load_model(panel,kind,plan):
    arch='base' if kind=='base' else 'q0'
    model,probe,head,rule,info=v.load(panel,arch)
    assert info['checkpoint_sha256']==plan['panels'][panel]['initial'][arch]['sha256']
    if kind in ['residual','zero']:
        # Same constructor/seed as legacy Stage2; copy only the new reader.
        torch.manual_seed(7919)
        fresh=b.PrefixPredictor(copy.deepcopy(model.base),model.base.inp[1].in_features,
            stage='updater',zero_residual=kind=='zero',checkpoint_blocks=True).cuda()
        with torch.no_grad():fresh.initial_tokens.copy_(model.initial_tokens)
        assert reader_hash(fresh)==reader_hash(model)
        model=fresh
    model.eval();probe.eval().requires_grad_(False);head.eval().requires_grad_(False)
    return model,probe,head,rule


def step(model,probe,head,rule,source,episodes,opt,sched,fast,verify=False):
    model.train();opt.zero_grad(set_to_none=True);events=[];active=False
    for ep in episodes:
        loss,event=(updater_loss if fast else p.updater_loss)(model,probe,head,rule,source,ep,verify=verify)
        assert torch.isfinite(loss)
        if loss.requires_grad:(loss/2).backward();active=True
        events.append(event);del loss
    if active:
        norm=torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.)
        assert torch.isfinite(norm) and norm>0
        assert all(x.grad is None for x in model.parameters() if not x.requires_grad)
        opt.step()
    sched.step()
    return {'events':events,'optimizer_step':active}


def optimizer(model):
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=3e-4,weight_decay=.05)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-4,total_steps=TOTAL,pct_start=.03)
    return opt,sched


def case_input(panel,task,plan):
    d,status,path=m.input_case(panel,task);spec=plan['panels'][panel]['cases'][task]
    assert sha(path)==spec['shortlist_sha256']
    assert sha(path.with_name(path.stem+'_labels.npy'))==spec['labels_sha256']
    return d,status


def baseline(panel,task,arch,plan):
    folder=Path(plan['panels'][panel]['initial'][arch]['evaluation'])/task.replace(':','_')
    path=folder/'scores.npz'
    assert sha(path)==plan['panels'][panel]['cases'][task][arch+'_scores_sha256']
    with np.load(path) as z:return {k:z[k] for k in z.files}


def preflight(panel):
    plan=plan_checked();prov=provenance();out=ROOT/'preflight'/panel
    out.mkdir(parents=True,exist_ok=True)
    done=out/'complete.json'
    if done.exists():assert read(done)['provenance']==prov;return
    source,rows=source_for(panel,plan);records={}
    episodes=[p.episode(rows,0,i) for i in range(16)]
    stress=[(row,qi,times,8) for row,qi,times,_ in v.stress_episodes(source,rows)]
    for kind in ['residual','zero']:
        model,probe,head,rule=load_model(panel,kind,plan)
        initial=copy.deepcopy(model.state_dict());frozen=reader_hash(model);ph=ident(probe)
        histories=[]
        for fast in [False,True]:
            model.load_state_dict(initial);model.zero_grad(set_to_none=True)
            torch.manual_seed(0);torch.cuda.manual_seed_all(0);opt,sched=optimizer(model);history=[]
            with v.controlled_backward('cuda',True):
                for i in range(9):
                    eps=episodes[2*i:2*i+2] if i<8 else stress
                    value,seconds=v.timed(lambda:step(model,probe,head,rule,source,eps,opt,sched,fast,True))
                    value.update(seconds=seconds,grad=v.fingerprint((n,x.grad) for n,x in model.named_parameters() if x.grad is not None),
                                 weights=ident(model))
                    history.append(value)
            assert reader_hash(model)==frozen and ident(probe)==ph
            assert all(e['used_updates']==8 for e in history[-1]['events'])
            histories.append(history);del opt,sched
        for a,c in zip(*histories):
            for key in ['events','optimizer_step','grad','weights']:assert a[key]==c[key],(kind,key)
        records[kind]=histories
        if kind=='zero':
            model.eval();r=torch.randn(1,32,196,128,device='cuda');a=torch.randn(1,3,device='cuda')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                np.testing.assert_array_equal(model.update_state(model.initial_state(),r,a).float().cpu().numpy(),
                                              model.update_state(model.initial_state(),r*11,a).float().cpu().numpy())
        del model,probe,head;gc.collect();torch.cuda.empty_cache()
    # New checkpoint + old loss replay verified above. Also check the real full
    # candidate pool against previously saved static predictions in default CUDA.
    model,probe,head,rule=load_model(panel,'q0',plan)
    task=plan['panels'][panel]['tasks'][0];d,status=case_input(panel,task,plan)
    z=torch.from_numpy(d['z0']).cuda().float();a=torch.from_numpy(d['action']).cuda();features=[f[None] for f in d['features']]
    ranked=CandidateRanker(model,head,rule,z,a,features).rank()
    np.testing.assert_array_equal(ranked.scores,baseline(panel,task,'q0',plan)['scores'])
    # A real training-pool failure sequence exercises the complete new TTA path.
    row,_,_,_=stress[0];ids=row['candidates'];z0,actions,feats=p.initial_data(source,ids)
    initial=ident(model);setting=plan['panels'][panel]['tta']
    trace=v.trajectory(model,head,rule,z0,actions,feats,
        lambda j:torch.from_numpy(source.z([ids[j]])).cuda().float(),'tta',True,
        steps=9,lr=setting['lr'],tta_steps=setting['steps'])
    assert ident(model)==initial
    write(out/'training_histories.json',records)
    write(out/'tta_history.json',trace)
    write(done,{'passed':True,'provenance':prov,'legacy_loss_events_gradients_weights_exact':True,
                'max_updates':8,'default_full_pool_Q0_scores_exact':True,'tta_reset_exact':True})
    print('PREFLIGHT_PASS',panel,flush=True)


def train(panel,kind):
    plan=plan_checked();prov=provenance()
    assert read(ROOT/'preflight'/panel/'complete.json')['provenance']==prov
    folder=ROOT/'models'/panel/kind;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        d=read(folder/'complete.json');assert d['provenance']==prov and sha(d['checkpoint'])==d['checkpoint_sha256'];return
    model,probe,head,rule=load_model(panel,kind,plan);source,rows=source_for(panel,plan)
    frozen=reader_hash(model);probe_hash=ident(probe);head_hash=ident(head);initial=ident(model)
    params=[n for n,x in model.named_parameters() if x.requires_grad]
    assert params and all(not n.startswith('base.') and n!='initial_tokens' for n in params)
    opt,sched=optimizer(model);start=updates=empty=supports=0;elapsed=0.;timings=[]
    resume=folder/'resume.pt';event_path=folder/'episodes.jsonl'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['provenance']==prov and ck['initial_sha256']==initial
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['slots'];updates=ck['optimizer_steps'];empty=ck['empty'];supports=ck['supports'];elapsed=ck['elapsed_s']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng']);timings=ck['timings']
    # Truncate only this run's uncommitted journal suffix after a failed attempt.
    if event_path.exists():
        keep=[]
        for line in event_path.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:break
            if r['slot']<start:keep.append(line)
        assert len(keep)==start
        event_path.write_text(''.join(line+'\n' for line in keep))
    else:assert start==0
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with event_path.open('a',buffering=1) as journal:
        for slot in range(start,TOTAL):
            eps=[p.episode(rows,0,2*slot+j) for j in range(2)]
            value,seconds=v.timed(lambda:step(model,probe,head,rule,source,eps,opt,sched,True,slot<3))
            updates+=int(value['optimizer_step']);empty+=sum(e['used_updates']==0 for e in value['events'])
            supports+=sum(e['used_updates'] for e in value['events']);timings.append(seconds);timings=timings[-1000:]
            value.update(slot=slot,seconds=seconds);journal.write(json.dumps(value)+'\n')
            done=slot+1
            if done%100==0 or done==TOTAL:
                progress={'slots':done,'total':TOTAL,'optimizer_steps':updates,'zero_support_episodes':empty,
                    'support_updates':supports,'updates_per_episode':supports/(2*done),
                    'elapsed_s':elapsed+time.perf_counter()-started,'recent_slot_mean_s':float(np.mean(timings)),
                    'projected_remaining_h':float(np.mean(timings))*(TOTAL-done)/3600,
                    'loss':float(np.mean([e['loss'] for e in value['events']])),
                    'lr':sched.get_last_lr()[0],'peak_GiB':torch.cuda.max_memory_allocated()/2**30,
                    'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'provenance':prov}
                write(folder/'progress.json',progress);print('TRAIN',panel,kind,json.dumps({k:x for k,x in progress.items() if k!='provenance'}),flush=True)
                if done%1000==0:write(folder/f'eta_{done:05d}.json',progress)
            if done%500==0 or done==TOTAL:
                assert reader_hash(model)==frozen and ident(probe)==probe_hash and ident(head)==head_hash
                journal.flush();os.fsync(journal.fileno())
                b.atomic_save(resume,{'provenance':prov,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
                    'scheduler':sched.state_dict(),'slots':done,'optimizer_steps':updates,'empty':empty,'supports':supports,
                    'elapsed_s':elapsed+time.perf_counter()-started,'initial_sha256':initial,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'timings':timings})
    checkpoint=folder/'model.pt'
    b.atomic_save(checkpoint,{'provenance':prov,'state_dict':model.state_dict(),'panel':panel,'kind':kind,'slots':TOTAL})
    write(folder/'complete.json',{'complete':True,'provenance':prov,'checkpoint':str(checkpoint),
        'checkpoint_sha256':sha(checkpoint),'initial_sha256':initial,'reader_sha256':frozen,
        'probe_sha256':probe_hash,'head_sha256':head_hash,'slots':TOTAL,'optimizer_steps':updates,
        'episodes':2*TOTAL,'zero_support_episodes':empty,'support_updates':supports,
        'elapsed_s':elapsed+time.perf_counter()-started,'episodes_sha256':sha(event_path)})


def audit_case(folder,plan):
    result=read(folder/'result.json')
    assert result['provenance']==provenance()
    panel,kind,task=result['panel'],result['kind'],result['task']
    d,status=case_input(panel,task,plan)
    with np.load(folder/'scores.npz') as z:
        scores=z['scores'];selected=z['selected'];indices=z['indices'];labels=z['labels']
    np.testing.assert_array_equal(indices,d['action_idx']);np.testing.assert_array_equal(labels,status>0)
    reference=baseline(panel,task,'base' if kind=='base' else 'q0',plan)
    np.testing.assert_array_equal(scores[0],reference['scores'])
    if kind in ['base','q0']:
        assert len(scores)==1 and len(selected)==0
        first=p.first_rank(scores[0],labels)
    else:
        assert 1<=len(selected)==len(scores)<=BUDGET and len(set(selected.tolist()))==len(selected)
        tried=np.zeros(len(labels),bool);first=None
        for turn,(values,j) in enumerate(zip(scores,selected),1):
            assert np.isfinite(values).all();masked=values.copy();masked[tried]=-np.inf
            assert int(np.argmax(masked))==j and not tried[j];tried[j]=True
            assert result['events'][turn-1]['action_idx']==int(indices[j])
            if labels[j]:assert turn==len(selected);first=turn
            elif turn<len(selected):
                assert result['events'][turn-1]['observation_action_idx']==int(indices[j])
                if kind=='tta':
                    assert result['events'][turn-1]['adaptation']['replay_indices']==selected[max(0,turn-5):turn].tolist()
        assert first is not None or len(selected)==BUDGET
    assert result['first_success']==first
    return result,scores,labels,selected


def evaluate(panel,kind,shard):
    plan=plan_checked();prov=provenance()
    assert read(ROOT/'preflight'/panel/'complete.json')['provenance']==prov
    model,probe,head,rule=load_model(panel,kind,plan)
    checkpoint_sha=plan['panels'][panel]['initial']['base' if kind=='base' else 'q0']['sha256']
    if kind in ['residual','zero']:
        ck=read(ROOT/'models'/panel/kind/'complete.json')
        assert ck['provenance']==prov and sha(ck['checkpoint'])==ck['checkpoint_sha256']
        model.load_state_dict(torch.load(ck['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
        checkpoint_sha=ck['checkpoint_sha256']
    model.eval().requires_grad_(False);initial_hash=ident(model)
    observations=m.Observations();tasks=plan['panels'][panel]['tasks'][shard::4]
    try:
        for number,task in enumerate(tasks):
            folder=ROOT/'evaluation'/panel/kind/task.replace(':','_');folder.mkdir(parents=True,exist_ok=True)
            if (folder/'result.json').exists():audit_case(folder,plan);continue
            d,status=case_input(panel,task,plan);z0=torch.from_numpy(d['z0']).cuda().float()
            actions=torch.from_numpy(d['action']).cuda();features=[f[None] for f in d['features']]
            setting=plan['panels'][panel]['tta']
            adapter=PredictorAdapter(model,lr=setting['lr'],steps=setting['steps']) if kind=='tta' else None
            state=model.initial_state().detach().clone() if kind in ['residual','zero'] else None
            ranker=CandidateRanker(model,head,rule,z0,actions,features)
            tried=np.zeros(1024,bool);scores=[];selected=[];events=[];states=[];success=None;start=time.perf_counter()
            for turn in range(BUDGET if kind not in ['base','q0'] else 1):
                ranked,rank_s=v.timed(lambda:ranker.rank(state,tried))
                if turn==0:np.testing.assert_array_equal(ranked.scores,baseline(panel,task,'base' if kind=='base' else 'q0',plan)['scores'])
                scores.append(ranked.scores)
                if state is not None:states.append(state.detach().float().cpu().numpy()[0])
                if kind in ['base','q0']:
                    success=p.first_rank(ranked.scores,status>0);events.append({'turn':1,'rank_s':rank_s});break
                j=ranked.selected;assert j is not None and not tried[j];tried[j]=True;selected.append(j)
                event={'turn':turn+1,'local':j,'action_idx':int(d['action_idx'][j]),'rank_s':rank_s,
                       'solved':bool(status[j]>0)};events.append(event)
                if status[j]>0:success=turn+1;break
                if turn+1==BUDGET:break
                cache=m.ASSETS/panel/'observations'/f'{task.replace(":","_")}_{int(d["action_idx"][j]):06d}.npz'
                event['observation_cache_hit']=cache.exists()
                truth,observation_s=v.timed(lambda:observations.get(panel,task,int(d['action_idx'][j]),
                    d['action'][j],d['z0'][j],int(status[j])))
                event.update(observation_s=observation_s,observation_action_idx=int(d['action_idx'][j]))
                if adapter:
                    adapted,seconds=v.timed(lambda:adapter.update(j,z0[j:j+1],actions[j:j+1],truth))
                    event.update(adaptation=adapted,update_s=seconds)
                else:
                    def update():
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                            return model.update_state(state,truth-ranked.prediction,actions[j:j+1])
                    state,seconds=v.timed(update);event['update_s']=seconds
            if adapter:adapter.reset()
            assert ident(model)==initial_hash
            p.f.save_npz(folder/'scores.npz',{'scores':np.stack(scores),'selected':np.array(selected,dtype=np.int64),
                'indices':d['action_idx'],'labels':status>0,'states':np.array(states,dtype=np.float32)})
            result={'panel':panel,'kind':kind,'task':task,'first_success':success,'events':events,
                'seconds':time.perf_counter()-start,'provenance':prov,'checkpoint_sha256':checkpoint_sha,
                'model_reset_exact':True}
            write(folder/'result.json',result);audit_case(folder,plan)
            write(ROOT/'progress'/f'eval_{panel}_{kind}_{shard}.json',{'complete_tasks':number+1,'total':len(tasks),'last_task':task})
            print('EVAL',panel,kind,shard,number+1,task,success,flush=True)
    finally:observations.close()
    write(ROOT/'evaluation'/panel/kind/f'shard_{shard}.json',{'complete':True,'tasks':tasks,'provenance':prov})


def summarize():
    plan=plan_checked();prov=provenance();summary={'complete':False,'raw_audit_passed':False,
        'provenance':prov,'panels':{},'training':{}}
    for panel,spec in plan['panels'].items():
        summary['panels'][panel]={};summary['training'][panel]={}
        for kind in ['residual','zero']:
            complete=read(ROOT/'models'/panel/kind/'complete.json')
            assert complete['provenance']==prov and complete['slots']==TOTAL
            assert sha(complete['checkpoint'])==complete['checkpoint_sha256']
            assert sha(ROOT/'models'/panel/kind/'episodes.jsonl')==complete['episodes_sha256']
            summary['training'][panel][kind]=complete
        assert summary['training'][panel]['residual']['initial_sha256']==summary['training'][panel]['zero']['initial_sha256']
        assert summary['training'][panel]['residual']['reader_sha256']==summary['training'][panel]['zero']['reader_sha256']
        allcases={}
        for kind in plan['methods']:
            rows=[];diagnostics=[]
            for task in spec['tasks']:
                folder=ROOT/'evaluation'/panel/kind/task.replace(':','_')
                rr,scores,labels,selected=audit_case(folder,plan);rows.append(rr)
                if len(scores)>1:
                    mask=np.ones(len(labels),bool);mask[selected[0]]=False
                    before=p.first_rank(scores[0,mask],labels[mask]);after=p.first_rank(scores[1,mask],labels[mask])
                    diagnostics.append({'task':task,'first_support_action_idx':rr['events'][0]['action_idx'],
                        'best_success_rank_before':before,'best_success_rank_after':after,
                        'auc_before':b.auc(labels[mask],scores[0,mask]),'auc_after':b.auc(labels[mask],scores[1,mask])})
            allcases[kind]={r['task']:r['first_success'] for r in rows}
            summary['panels'][panel][kind]={'n':len(rows),'counts':{f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in rows) for k in [1,3,5,9]},
                'first_success':allcases[kind],'first_update':diagnostics,'seconds':sum(r['seconds'] for r in rows)}
        for kind in ['residual','zero','tta']:
            for budget in [1,3,5,9]:
                solved=lambda x:x is not None and x<=budget
                gained=[t for t in spec['tasks'] if solved(allcases[kind][t]) and not solved(allcases['q0'][t])]
                lost=[t for t in spec['tasks'] if not solved(allcases[kind][t]) and solved(allcases['q0'][t])]
                summary['panels'][panel][kind].setdefault('paired_Q0',{})[str(budget)]={'gained':gained,'lost':lost,'delta_count':len(gained)-len(lost)}
    summary.update(complete=True,raw_audit_passed=True);write(ROOT/'SUMMARY.json',summary)
    lines=['# Milestone memory results','', 'Same Q0 start, legacy terminal-state loss, one seed; 100 scenes per panel, 9 attempts.','',
        '| Panel | Method | S@1 | S@3 | S@5 | S@9 | Gained/lost vs Q0 at 9 |','|---|---|---:|---:|---:|---:|---|']
    for panel,arms in summary['panels'].items():
        for kind,d in arms.items():
            c=d['counts'];paired=d.get('paired_Q0',{}).get('9');pair=f'{len(paired["gained"])}/{len(paired["lost"])}' if paired else '—'
            lines.append(f'| {panel} | {kind} | {c["s1"]}% | {c["s3"]}% | {c["s5"]}% | {c["s9"]}% | {pair} |')
    lines+=['','All raw action choices, initial scores, success stopping and paired counts audited.','',
        'Full timing, first-update ranking diagnostics and task IDs are in SUMMARY.json. This compares methods on the existing shared shortlist; no universal or multi-seed claim.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',ROOT/'SUMMARY.json',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','train','eval','summary'])
    ap.add_argument('--panel',choices=['cross','within']);ap.add_argument('--kind',choices=['base','q0','residual','zero','tta']);ap.add_argument('--shard',type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0)
    if args.mode=='prepare':prepare()
    elif args.mode=='preflight':preflight(args.panel)
    elif args.mode=='train':train(args.panel,args.kind)
    elif args.mode=='eval':evaluate(args.panel,args.kind,args.shard)
    else:summarize()


if __name__=='__main__':main()
