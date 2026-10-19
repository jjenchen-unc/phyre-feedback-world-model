"""Q0-first feedback pilot; same pools, fresh updater training, matched evaluation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

import selector_feedback_data as pools
data=pools.data
R=pools.R
import deploy_pool_updater as prior
prior.R=R
old=prior.old
from fast_inference import PreparedScorer, CandidateRanker
from memory_geometry_followup import geometry_loss
from multiturn_runtime import PredictorAdapter
from updater_validation_numerics import controlled_backward


class LocalEncoder:
    """Canonical full observations on A6000, without a remote encoder service."""
    def __init__(self):
        assert 'A6000' in torch.cuda.get_device_name(0)
        self.encoder=old.m.Encoder(old.m.PCA)
    def initial(self,frames):return self.encoder.initial(frames)
    def rollout(self,frames):
        result,self.last_rollout_s=timed(lambda:self.encoder.rollout(frames))
        return result
    def close(self):del self.encoder;torch.cuda.empty_cache()


class TimedTruthCache(data.TruthCache):
    def get(self,task,local,pool):
        key=f'{task.replace(":","_")}_{int(pool["action_idx"][local]):06d}'
        path=self.root/str(int(hashlib.sha256(key.encode()).hexdigest(),16)%32)/(key+'.npz')
        hit=path.exists();start=time.time()
        physical=dict(pool,action=pool['simulation_action'])
        value=super().get(task,local,physical)
        value.update(cache_hit=hit,total_observation_s=time.time()-start,
                     observation_encode_s=0. if hit else getattr(self.encoder,'last_rollout_s',0.))
        return value


def provenance():
    return {'plan':data.sha(R/'PLAN.json'),
            'code':{f.name:data.sha(f) for f in sorted((R/'code').glob('*.py'))}}


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();v=fn();torch.cuda.synchronize()
    return v,time.perf_counter()-start


class Team(prior.Team):
    def __init__(self,panel,kind,plan):
        self.rank=0;self.world=1;self.panel=panel;self.kind=kind;self.plan=plan
        self.model,self.probe,self.head,self.rule=old.load_model(panel,kind,plan)
        self.model.checkpoint_blocks=False;self.model.eval()
        self.scorer=PreparedScorer(self.head,self.rule)
        self.truth=TimedTruthCache(panel);self.chunks={};self.pool=None;self.task=None
    def request(self,**command):return [self.execute(command)]
    def update(self,state,j,observation=None):
        action=torch.from_numpy(self.pool['action'][j:j+1]).cuda()
        if self.kind=='zero':
            residual=torch.zeros(1,32,196,128,device='cuda')
        else:
            pred=self.predict(state,j)[j%32:j%32+1]
            residual=torch.from_numpy(observation['latent'][None]).cuda().float()-pred
        with torch.autocast('cuda',dtype=torch.bfloat16):return self.model.update_state(state,residual,action)
    def collect(self,index,verify=False,cap_override=None):
        task,cap,perm,times=prior.draw_episode(self.plan['panels'][self.panel]['train_tasks'],index)
        if cap_override is not None:cap=cap_override
        self.load(task);state=self.model.initial_state().detach().clone()
        selected=[];supports=[];solved=[];obs={};tried=np.zeros(1024,bool);start=time.time()
        first_scores=None
        for turn in range(cap+1):
            scores=self.scores(state,verify=verify and turn in [0,1])
            if turn==0:
                first_scores=scores.copy()
                if verify:
                    ranker=CandidateRanker(self.model,self.head,self.rule,
                        torch.from_numpy(self.pool['z0'].copy()).cuda().float(),
                        torch.from_numpy(self.pool['action']).cuda(),[f[None] for f in self.pool['features']])
                    ranked=ranker.rank(state,tried)
                    np.testing.assert_array_equal(ranked.scores,scores)
                    np.testing.assert_array_equal(ranker.rank(None,tried).scores,scores)
                    del ranker,ranked
            j=pools.choose_trial(scores,tried,turn)
            assert not tried[j];tried[j]=True;selected.append(j);solved.append(bool(self.pool['status'][j]>0))
            if solved[-1] or turn==cap:break
            if self.kind!='zero':obs[j]=self.observations([j])[j]
            with torch.no_grad():state=self.update(state,j,obs.get(j))
            supports.append(j)
        qi=prior.query_indices(perm,selected)
        assert selected[0]==int(first_scores.argmax()) and not set(selected)&set(qi)
        assert not {tuple(self.pool['action'][j]) for j in selected}&{tuple(self.pool['action'][j]) for j in qi}
        if supports:obs.update(self.observations(qi))
        return {'task':task,'cap':cap,'selected':selected,'supports':supports,'solved':solved,
                'query':qi,'times':times,'observations':obs,'final_state':state,'first_scores':first_scores,'collection_s':time.time()-start}
    def event(self,ep,loss,l1,geometry):
        event=super().event(ep,loss,l1,geometry)
        event.update(first_rank_source='q0_predictor',
            first_q0_argmax=int(self.pool['action_idx'][int(ep['first_scores'].argmax())]),
            classifier_first_action=int(self.pool['action_idx'][0]))
        assert event['selected'][0]==event['first_q0_argmax']
        return event
    def loss(self,ep,whole_checkpoint=True,verify=False):
        state=self.model.initial_state();obs=ep['observations']
        if not ep['supports']:return torch.zeros((),device='cuda'),None,None
        for j in ep['supports']:
            if self.kind=='zero':state=self.update(state,j)
            else:
                z,a=self.chunk(j//32*32)
                def forward(initial,actions,current):
                    with torch.autocast('cuda',dtype=torch.bfloat16):return self.model(initial,actions,current.expand(len(actions),-1,-1))
                pred=checkpoint(forward,z,a,state,use_reentrant=False) if whole_checkpoint else forward(z,a,state)
                residual=torch.from_numpy(obs[j]['latent'][None]).cuda().float()-pred[j%32:j%32+1]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    state=self.model.update_state(state,residual,torch.from_numpy(self.pool['action'][j:j+1]).cuda())
        if verify:assert torch.equal(state.detach(),ep['final_state'])
        qi=ep['query'];z0=torch.from_numpy(self.pool['z0'][qi].copy()).cuda().float()
        target=torch.from_numpy(np.stack([obs[j]['latent'] for j in qi])).cuda().float()
        actions=torch.from_numpy(self.pool['action'][qi]).cuda()
        def terms(current):
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.model(z0,actions,current.expand(8,-1,-1)).float()
            l1=(pred[:,1:]-target[:,1:]).abs().mean()
            geo=geometry_loss(self.probe,pred,[obs[j]['features'] for j in qi],ep['times'])
            return l1+.1*geo,l1.detach(),geo.detach()
        return checkpoint(terms,state,use_reentrant=False) if whole_checkpoint else terms(state)


def optimizer(model,total):
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=.05)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-4,total_steps=total,pct_start=.03)
    return opt,sched


def preflight(team):
    initial=copy.deepcopy(team.model.state_dict());reader=old.reader_hash(team.model)
    probe=old.ident(team.probe);head=old.ident(team.head);records=[];gradient=False
    opt,_=optimizer(team.model,team.plan['slots'])
    torch.cuda.reset_peak_memory_stats();start=time.time()
    for index in range(8):
        ep=team.collect(index,True,cap_override=1 if index<2 else 8)
        assert ep['selected'][0]==int(ep['first_scores'].argmax())
        if ep['supports'] and not gradient:
            tests=[]
            for whole in [False,True]:
                team.model.zero_grad(set_to_none=True);team.model.checkpoint_blocks=not whole
                with controlled_backward('cpu',True):
                    loss,l1,geo=team.loss(ep,whole,True);loss.backward()
                tests.append((float(loss),old.v.fingerprint((n,p.grad) for n,p in team.model.named_parameters() if p.grad is not None)))
                del loss
            assert tests[0]==tests[1];gradient=True;team.model.checkpoint_blocks=False
        opt.zero_grad(set_to_none=True)
        loss,l1,geo=team.loss(ep,True,True)
        if loss.requires_grad:
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in team.model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(norm) and norm>0;opt.step()
        records.append(team.event(ep,loss,l1,geo));del loss,ep
        if gradient and index>=3:break
    assert gradient
    # Explicit stress-only replay of eight real failures, not training sampler selection.
    failed=np.flatnonzero(team.pool['status']<0)[:8].tolist();assert len(failed)==8
    qi=[j for j in range(1024) if j not in failed][:8]
    obs=team.observations(qi+(failed if team.kind!='zero' else []))
    state=team.model.initial_state().detach().clone()
    with torch.no_grad():
        for j in failed:state=team.update(state,j,obs.get(j))
    ep={'supports':failed,'query':qi,'observations':obs,'final_state':state,'times':[list(range(1,32,4))]*8}
    team.model.zero_grad(set_to_none=True);loss,_,_=team.loss(ep,True,True);loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in team.model.parameters())
    if team.kind=='zero':
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            a=torch.randn(1,3,device='cuda');r=torch.randn(1,32,196,128,device='cuda')
            assert torch.equal(team.model.update_state(state,r,a),team.model.update_state(state,torch.zeros_like(r),a))
    # TTA must change permitted weights, then restore them exactly, on a real failed action.
    tta_model,_,_,_=old.load_model(team.panel,'q0',team.plan)
    tta_initial=old.ident(tta_model);setting=team.plan['panels'][team.panel]['tta']
    adapter=PredictorAdapter(tta_model,lr=setting['lr'],steps=setting['steps'],replay=setting['replay'])
    j=failed[0];truth=team.observations([j])[j]
    adapter.update(j,torch.from_numpy(team.pool['z0'][j:j+1].copy()).cuda().float(),
                   torch.from_numpy(team.pool['action'][j:j+1]).cuda(),torch.from_numpy(truth['latent'][None]).cuda().float())
    assert old.ident(tta_model)!=tta_initial;adapter.reset();assert old.ident(tta_model)==tta_initial
    del adapter,tta_model
    assert old.reader_hash(team.model)==reader and old.ident(team.probe)==probe and old.ident(team.head)==head
    team.model.load_state_dict(initial);team.model.zero_grad(set_to_none=True)
    if team.kind=='residual':
        if team.truth.encoder is None:team.truth.encoder=LocalEncoder()
        renderer=pools.coverage.Renderer()
        try:
            raw,render_s=timed(lambda:renderer.render(team.task,team.pool['action_idx']))
            np.testing.assert_array_equal(raw['indices'],team.pool['action_idx'])
            encoded,encode_s=timed(lambda:team.truth.encoder.initial(raw['frames']))
            np.testing.assert_array_equal(encoded,team.pool['z0'])
        finally:renderer.close()
        with np.load(R/'inputs'/(team.panel+'_train_scenes.npz')) as z:
            frame=z['frames'][z['tasks'].tolist().index(team.task):][:1]
        enc=team.truth.encoder.encoder
        def initial_scene_encode():
            with torch.no_grad():
                pixels=enc.pixels(frame)
                return enc.model(pixel_values=torch.stack([pixels,pixels],2)).last_hidden_state
        _,initial_s=timed(initial_scene_encode)
        data.write(R/'cost'/f'{team.panel}_input.json',{'provenance':provenance(),'gpu':torch.cuda.get_device_name(0),
                   'single_scene_encode_s':initial_s,'render1024_s':render_s,'encode1024_s':encode_s,
                   'encoded_z0_exact':True,'model_loading_excluded':True,'scope':'One training scene; latency probe, not population estimate'})
    # A short natural training timing, separate from expensive equivalence/stress checks.
    pilot=[]
    for slot in range(2):
        clock=time.time();team.model.zero_grad(set_to_none=True);events=[]
        for offset in range(2):
            ep=team.collect(2*slot+offset);loss,l1,geo=team.loss(ep,True,True)
            if loss.requires_grad:(loss/2).backward()
            events.append(team.event(ep,loss,l1,geo));del loss,ep
        pilot.append({'seconds':time.time()-clock,'events':events})
    team.model.load_state_dict(initial);team.model.zero_grad(set_to_none=True)
    data.write(R/'preflight'/team.panel/team.kind/'complete.json',{
        'passed':True,'provenance':provenance(),'q0_first':True,'training_eval_initial_scores_exact':True,'query_disjoint':True,
        'current_residual_replay_exact':True,'checkpoint_gradient_exact':True,'stress8_real_failures':True,
        'reader_probe_head_frozen':True,'tta_update_and_reset_verified':True,
        'records':records,'natural_cost_pilot':pilot,'pilot_limit':'Two slots, some hot truth caches; not a reliable full-training ETA',
        'seconds':time.time()-start,
        'peak_GiB':torch.cuda.max_memory_allocated()/2**30})
    print('PREFLIGHT_PASS',team.panel,team.kind,flush=True)


def train(team):
    prov=provenance();folder=R/'models'/team.panel/team.kind;folder.mkdir(parents=True,exist_ok=True)
    assert data.read(R/'DATA_AUDIT.json')['plan']==prov['plan']
    # Every arm must pass before any formal training.
    for panel in team.plan['panels']:
        for kind in ['residual','zero']:
            assert data.read(R/'preflight'/panel/kind/'complete.json')['provenance']==prov
    total=team.plan['slots'];model=team.model;initial=old.ident(model);reader=old.reader_hash(model)
    opt,sched=optimizer(model,total);resume=folder/'resume.pt';start=updates=0;elapsed=0.
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['provenance']==prov
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['slots'];updates=ck['optimizer_steps'];elapsed=ck['elapsed_s']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
    journal=folder/'episodes.jsonl'
    if journal.exists():
        rows=[json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
        rows=[r for r in rows if r['slot']<start];assert len(rows)==start
        journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    began=time.time();torch.cuda.reset_peak_memory_stats()
    with journal.open('a',buffering=1) as f:
        for slot in range(start,total):
            clock=time.time();opt.zero_grad(set_to_none=True);active=False;events=[]
            for offset in range(2):
                ep=team.collect(2*slot+offset);loss,l1,geo=team.loss(ep,True,slot<3)
                assert torch.isfinite(loss)
                if loss.requires_grad:(loss/2).backward();active=True
                events.append(team.event(ep,loss,l1,geo));del loss,ep
            if active:
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                assert torch.isfinite(norm) and norm>0;opt.step();updates+=1
            sched.step();done=slot+1
            f.write(json.dumps({'slot':slot,'optimizer_step':active,'events':events,'seconds':time.time()-clock})+'\n')
            if done%8==0 or done==total:
                assert old.reader_hash(model)==reader
                progress={'slots':done,'total':total,'optimizer_steps':updates,'elapsed_s':elapsed+time.time()-began,
                          'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'provenance':prov}
                data.write(folder/'progress.json',progress);print('TRAIN',team.panel,team.kind,done,progress['elapsed_s'],flush=True)
            if done%32==0 or done==total:
                f.flush();os.fsync(f.fileno())
                old.b.atomic_save(resume,{'provenance':prov,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
                    'scheduler':sched.state_dict(),'slots':done,'optimizer_steps':updates,'elapsed_s':elapsed+time.time()-began,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
    path=folder/'model.pt';old.b.atomic_save(path,{'state_dict':model.state_dict(),'provenance':prov})
    data.write(folder/'complete.json',{'complete':True,'provenance':prov,'checkpoint':str(path),'checkpoint_sha256':data.sha(path),
        'initial_sha256':initial,'reader_sha256':reader,'slots':total,'optimizer_steps':updates,'elapsed_s':elapsed+time.time()-began})


def evaluate(panel,kind,shard,nshards):
    plan=data.checked();prov=provenance();team=Team(panel,kind,plan)
    if kind in ['residual','zero']:
        ck=data.read(R/'models'/panel/kind/'complete.json');assert ck['provenance']==prov
        assert data.sha(ck['checkpoint'])==ck['checkpoint_sha256']
        team.model.load_state_dict(torch.load(ck['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
    team.model.eval().requires_grad_(False);initial=old.ident(team.model)
    try:
        for task in plan['panels'][panel]['tasks'][shard::nshards]:
            folder=R/'evaluation'/panel/kind/task.replace(':','_')
            if (folder/'result.json').exists():assert data.read(folder/'result.json')['provenance']==prov;continue
            team.load(task);pool=team.pool;z0=torch.from_numpy(pool['z0'].copy()).cuda().float()
            actions=torch.from_numpy(pool['action']).cuda();features=[f[None] for f in pool['features']]
            ranker=CandidateRanker(team.model,team.head,team.rule,z0,actions,features)
            adapter=None
            if kind=='tta':
                setting=plan['panels'][panel]['tta'];adapter=PredictorAdapter(team.model,lr=setting['lr'],steps=setting['steps'],replay=setting['replay'])
            state=team.model.initial_state().detach().clone();tried=np.zeros(1024,bool)
            events=[];history=[];selected=[];fixed=None;success=None;began=time.time()
            torch.cuda.reset_peak_memory_stats()
            for turn in range(plan['budget']):
                event={'turn':turn+1,'rank_s':0.,'update_s':0.,'observation_s':0.}
                if kind=='q0' and fixed is not None:values=fixed.copy()
                else:
                    ranked,event['rank_s']=timed(lambda:ranker.rank(state if kind in ['residual','zero'] else None,tried))
                    values=ranked.scores
                    if kind=='q0':fixed=values.copy()
                j=pools.choose_trial(values,tried,turn)
                event['rank_source']='q0_initial_predictor' if turn==0 else 'q0_or_updated_predictor'
                assert not tried[j];tried[j]=True;selected.append(j);history.append(values)
                event.update(action_idx=int(pool['action_idx'][j]),local=j,solved=bool(pool['status'][j]>0));events.append(event)
                if event['solved']:success=turn+1;break
                if turn+1==plan['budget']:break
                if kind in ['residual','tta']:
                    obs,event['observation_s']=timed(lambda:team.observations([j])[j])
                    event.update(observation_cache_hit=obs['cache_hit'],observation_encode_s=obs['observation_encode_s'])
                    if kind=='tta':
                        truth=torch.from_numpy(obs['latent'][None]).cuda().float()
                        adapted,event['update_s']=timed(lambda:adapter.update(j,z0[j:j+1],actions[j:j+1],truth))
                        event['adaptation']=adapted
                    else:
                        with torch.no_grad():state,event['update_s']=timed(lambda:team.update(state,j,obs))
                elif kind=='zero':
                    with torch.no_grad():state,event['update_s']=timed(lambda:team.update(state,j))
            if adapter:adapter.reset()
            assert old.ident(team.model)==initial
            assert selected[0]==int(history[0].argmax()) and len(set(selected))==len(selected)
            for turn,j in enumerate(selected):
                values=history[turn].copy();values[selected[:turn]]=-np.inf;assert int(values.argmax())==j
            data.save_npz(folder/'scores.npz',scores=np.stack(history),selected=np.array(selected),indices=pool['action_idx'],labels=pool['status']>0)
            data.write(folder/'result.json',{'provenance':prov,'panel':panel,'kind':kind,'task':task,
                'selector':plan['panels'][panel]['selector'],'first_success':success,'events':events,
                'seconds_excluding_initial100k_and_pool_build':time.time()-began,
                'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'model_reset_exact':True})
            print('EVAL',panel,kind,task,success,flush=True)
    finally:team.truth.close()
    data.write(R/'evaluation'/panel/kind/f'shard_{shard}.json',{'complete':True,'provenance':prov})


def summary():
    plan=data.checked();prov=provenance();out={'complete':True,'provenance':prov,'scope':'Q0-first;64 train scenes,256 slots,seed0 pilot','panels':{}}
    for panel,spec in plan['panels'].items():
        arms={};firsts={}
        for kind in plan['methods']:
            rows=[]
            for task in spec['tasks']:
                folder=R/'evaluation'/panel/kind/task.replace(':','_');row=data.read(folder/'result.json')
                assert row['provenance']==prov and row['model_reset_exact'];pool=data.load_pool(panel,task)
                with np.load(folder/'scores.npz') as z:
                    np.testing.assert_array_equal(z['indices'],pool['action_idx']);np.testing.assert_array_equal(z['labels'],pool['status']>0)
                    selected=z['selected'];assert selected[0]==int(z['scores'][0].argmax()) and len(set(selected))==len(selected)
                    for t,j in enumerate(selected):
                        sc=z['scores'][t].copy();sc[selected[:t]]=-np.inf;assert int(sc.argmax())==j
                    hits=np.flatnonzero(z['labels'][selected]);actual=int(hits[0]+1) if len(hits) else None
                    assert actual==row['first_success']
                    assert (actual==len(selected)) if actual is not None else (len(selected)==plan['budget'])
                firsts[(kind,task)]=row['events'][0]['action_idx'];rows.append(row)
            arms[kind]={'n':len(rows),'counts':{f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in rows) for k in [1,3,5,9]},
                'mean_feedback_path_s':float(np.mean([r['seconds_excluding_initial100k_and_pool_build'] for r in rows])),
                'rank_s':sum(e['rank_s'] for r in rows for e in r['events']),
                'update_s':sum(e['update_s'] for r in rows for e in r['events']),
                'observation_s':sum(e['observation_s'] for r in rows for e in r['events']),
                'first_success':{r['task']:r['first_success'] for r in rows}}
        assert all(len({firsts[(m,t)] for m in plan['methods']})==1 for t in spec['tasks'])
        static={}
        for task in spec['tasks']:
            hits=np.flatnonzero(data.load_pool(panel,task)['status']>0)
            static[task]=int(hits[0]+1) if len(hits) and hits[0]<plan['budget'] else None
        arms['selector_static']={'n':len(static),'counts':{f's{k}':sum(v is not None and v<=k for v in static.values()) for k in [1,3,5,9]},
                                 'first_success':static,'rank_s_after_first_attempt':0.,'update_s':0.,
                                 'source':'Original selector pool order; no Q0 reranking, no training'}
        costs={}
        for method in ['base','scene']:
            receipts=[data.read(f) for f in (pools.C/'scores'/panel/method).glob('*/complete.json')]
            sec=np.array([r['seconds_this_invocation'] for r in receipts])
            costs[method]={'mean_s':float(sec.mean()),'median_s':float(np.median(sec)),
                           'p90_s':float(np.quantile(sec,.9)),'allocated_gpu_hours':float(sec.sum()/3600),
                           'gpus':sorted(set(r['gpu'] for r in receipts)),'n_tasks':len(receipts)}
        training={k:data.read(R/'models'/panel/k/'complete.json') for k in ['residual','zero']}
        out['panels'][panel]={'selector':spec['selector'],'selector_decision':spec['selector_decision'],'methods':arms,
                              'full100k_rank_cost':costs,'input_latency_probe':data.read(R/'cost'/f'{panel}_input.json'),
                              'training':training}
    out['timing_limits']=['Initial100k timing from coverage run excludes classifier initial-scene encoding; encoding throughput recorded separately.',
       'Feedback wall times include whichever immutable truth-cache hits occurred; not a cold deployment latency claim.',
       'Allocated GPU time is not the same as CUDA-active kernel time. Single-GPU A6000 feedback path measured.']
    data.write(R/'SUMMARY.json',out)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','train','eval','summary'])
    ap.add_argument('--panel',choices=['cross','within']);ap.add_argument('--kind',choices=['q0','residual','zero','tta'])
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=1);a=ap.parse_args()
    torch.set_num_threads(2)
    data.EncoderProxy=LocalEncoder  # Training/evaluation run ONLY on canonical A6000.
    if a.mode=='summary':summary()
    elif a.mode=='eval':evaluate(a.panel,a.kind,a.shard,a.nshards)
    else:
        team=Team(a.panel,a.kind,data.checked())
        try:preflight(team) if a.mode=='preflight' else train(team)
        finally:team.truth.close()
