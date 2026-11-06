"""Current-policy updater training on deployment-style pools, distributed ranking.

Only rank 0 owns the trainable updater/optimizer. Other GPUs hold the identical
frozen reader, rank original 32-action chunks with current broadcast tokens, and
encode requested true observations. They never select supports or use stale
updater weights. Full BPTT and the effective two-episode optimizer batch remain.
"""
import argparse
import ast
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
import deploy_pool_data as data

R = data.R
old = data.runtime()
from fast_inference import PreparedScorer
from memory_geometry_followup import geometry_loss
from updater_validation_numerics import controlled_backward


def provenance():
    return {'plan':data.sha(R/'PLAN.json'),
            'code':{f.name:data.sha(f) for f in sorted((R/'code').glob('*.py'))}}


def equivalent_scores(actual,reference):
    np.testing.assert_allclose(actual,reference,rtol=0,atol=1e-14)
    # Tolerance is permitted only for CPU float64 arithmetic, never for changed
    # action ordering. All candidate ranks, including ties, must remain exact.
    np.testing.assert_array_equal(np.argsort(-np.asarray(actual),kind='stable'),
                                  np.argsort(-np.asarray(reference),kind='stable'))


def install_evaluation_guards():
    source=Path(old.__file__)
    assert data.sha(source)==data.checked()['dependencies'][str(source)]
    tree=ast.parse(source.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['audit_case','evaluate']]
    class ScoreGuard(ast.NodeTransformer):
        changed=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if ast.unparse(node.func)=='np.testing.assert_array_equal' and any('scores' in ast.unparse(a) for a in node.args):
                node.func=ast.Name(id='_deploy_equal_scores',ctx=ast.Load());self.changed+=1
            return node
    rewrite=ScoreGuard();module=ast.Module(body=functions,type_ignores=[]);rewrite.visit(module)
    assert rewrite.changed==2,rewrite.changed
    old.__dict__['_deploy_equal_scores']=equivalent_scores
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),old.__dict__)


def draw_episode(tasks, index):
    rng = np.random.default_rng([20261008,0,index,0])
    task = tasks[int(rng.integers(len(tasks)))]
    cap = int(rng.choice([1,2,4,8]))
    # Random permutation is outcome-independent; skip tried IDs after rollout.
    permutation = rng.permutation(1024)
    times = [sorted(rng.choice(np.arange(1,32),8,replace=False).tolist()) for _ in range(8)]
    return task, cap, permutation, times


def query_indices(permutation, selected):
    tried = set(selected)
    result = [int(j) for j in permutation if int(j) not in tried][:8]
    assert len(result)==8 and not tried.intersection(result)
    return result


class Team:
    def __init__(self,panel,kind,plan):
        self.rank=dist.get_rank();self.world=dist.get_world_size()
        self.panel=panel;self.kind=kind;self.plan=plan;self.pool=None;self.task=None
        self.model,self.probe,self.head,self.rule=old.load_model(panel,kind,plan)
        self.model.checkpoint_blocks=False;self.model.eval()
        self.scorer=PreparedScorer(self.head,self.rule)
        self.truth=data.TruthCache(panel)
        self.chunks={}
    def execute(self,command):
        mode=command['mode']
        if mode=='pool':
            self.task=command['task']
            if command.get('canonical',False):
                self.pool,status,_=old.m.input_case(self.panel,self.task)
                self.pool['status']=status
            else:self.pool=data.load_pool(self.panel,self.task,command.get('validation',False))
            self.chunks={}
            for start in range(0,1024,32):
                if (start//32)%self.world != self.rank:continue
                self.chunks[start]=(torch.from_numpy(self.pool['z0'][start:start+32].copy()).cuda().float(),
                                    torch.from_numpy(self.pool['action'][start:start+32]).cuda())
            return None
        if mode=='rank':
            state=command['state'].cuda();result=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for start,(z0,actions) in self.chunks.items():
                    pred=self.model(z0,actions,state.expand(32,-1,-1))
                    scores=self.scorer(pred,[f[None] for f in self.pool['features'][start:start+32]])
                    result.append((start,scores.cpu().numpy()))
            return result
        if mode=='truth':
            result=[]
            # Keep the large encoder off the BPTT GPU when helpers are available.
            workers=max(self.world-1,1)
            for position,j in enumerate(command['indices']):
                owner=1+position%workers if self.world>1 else 0
                if self.rank==owner:result.append(self.truth.get(self.task,j,self.pool))
            return result
        if mode=='stop':return None
        raise ValueError(mode)
    def request(self,**command):
        assert self.rank==0
        dist.broadcast_object_list([command],src=0)
        result=self.execute(command);gathered=[None]*self.world
        dist.gather_object(result,gathered,dst=0)
        return gathered
    def serve(self):
        while True:
            message=[None];dist.broadcast_object_list(message,src=0);command=message[0]
            result=self.execute(command);dist.gather_object(result,None,dst=0)
            if command['mode']=='stop':break
    def load(self,task,validation=False):
        self.request(mode='pool',task=task,validation=validation)
    def scores(self,state,verify=False):
        results=self.request(mode='rank',state=state.detach().cpu())
        scores=np.full(1024,np.nan)
        for records in results:
            for start,values in records:
                assert np.isnan(scores[start:start+32]).all();scores[start:start+32]=values
        assert np.isfinite(scores).all()
        if verify:
            reference=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for start in range(0,1024,32):
                    z,a=self.chunk(start)
                    pred=self.model(z,a,state.expand(32,-1,-1))
                    reference.append(self.scorer(pred,[f[None] for f in self.pool['features'][start:start+32]]).cpu().numpy())
            np.testing.assert_array_equal(scores,np.concatenate(reference))
        return scores
    def observations(self,indices):
        result={v['local']:v for items in self.request(mode='truth',indices=indices) for v in items}
        assert set(result)==set(indices)
        return result
    def chunk(self,start):
        if start in self.chunks:return self.chunks[start]
        return (torch.from_numpy(self.pool['z0'][start:start+32].copy()).cuda().float(),
                torch.from_numpy(self.pool['action'][start:start+32]).cuda())
    def predict(self,state,chosen):
        z,a=self.chunk(chosen//32*32)
        with torch.autocast('cuda',dtype=torch.bfloat16):return self.model(z,a,state.expand(32,-1,-1))
    def collect(self,index,verify=False,cap_override=None):
        task,cap,permutation,times=draw_episode(self.plan['panels'][self.panel]['train_tasks'],index)
        if cap_override is not None:cap=cap_override
        self.load(task)
        state=self.model.initial_state().detach().clone();selected=[];supports=[];solved=[];observations={};tried=np.zeros(1024,bool)
        t0=time.time();first_scores=None
        for turn in range(cap+1):
            scores=self.scores(state,verify=verify and turn in [0,1])
            if turn==0:first_scores=scores
            eligible=scores.copy();eligible[tried]=-np.inf;j=int(np.argmax(eligible))
            assert not tried[j];tried[j]=True;selected.append(j)
            success=bool(self.pool['status'][j]>0);solved.append(success)
            if success or turn==cap:break
            observation=self.observations([j])[j];observations[j]=observation
            truth=torch.from_numpy(observation['latent'][None]).cuda().float()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=self.predict(state,j)[j%32:j%32+1]
                action=torch.from_numpy(self.pool['action'][j:j+1]).cuda()
                state=self.model.update_state(state,truth-pred,action)
            supports.append(j)
        qi=query_indices(permutation,selected)
        assert not {tuple(self.pool['action'][j]) for j in selected}&{tuple(self.pool['action'][j]) for j in qi}
        if supports:observations.update(self.observations(qi))
        return {'task':task,'cap':cap,'selected':selected,'supports':supports,'solved':solved,
                'query':qi,'times':times,'observations':observations,'final_state':state,
                'first_scores':first_scores,'collection_s':time.time()-t0}
    def loss(self,episode,whole_checkpoint=True,verify=False):
        state=self.model.initial_state();supports=episode['supports'];obs=episode['observations']
        if not supports:return torch.zeros((),device='cuda'),None,None
        for j in supports:
            z,a=self.chunk(j//32*32)
            def forward(initial,action,current):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    return self.model(initial,action,current.expand(len(action),-1,-1))
            pred=checkpoint(forward,z,a,state,use_reentrant=False) if whole_checkpoint else forward(z,a,state)
            truth=torch.from_numpy(obs[j]['latent'][None]).cuda().float()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                state=self.model.update_state(state,truth-pred[j%32:j%32+1],
                                             torch.from_numpy(self.pool['action'][j:j+1]).cuda())
        if verify:assert torch.equal(state.detach(),episode['final_state']), 'Current-residual replay differs'
        qi=episode['query'];z0=torch.from_numpy(self.pool['z0'][qi].copy()).cuda().float()
        target=torch.from_numpy(np.stack([obs[j]['latent'] for j in qi])).cuda().float()
        actions=torch.from_numpy(self.pool['action'][qi]).cuda()
        features=[obs[j]['features'] for j in qi];times=episode['times']
        def query_loss(current):
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.model(z0,actions,current.expand(8,-1,-1)).float()
            l1=(pred[:,1:]-target[:,1:]).abs().mean()
            geometry=geometry_loss(self.probe,pred,features,times)
            return l1+.1*geometry,l1.detach(),geometry.detach()
        return checkpoint(query_loss,state,use_reentrant=False) if whole_checkpoint else query_loss(state)
    def event(self,episode,loss,l1,geometry):
        ids=self.pool['action_idx']
        return {'task':episode['task'],'cap':episode['cap'],'candidate_count':1024,
                'candidate_positive_fraction':float((self.pool['status']>0).mean()),
                'selected':[int(ids[j]) for j in episode['selected']],
                'supports':[int(ids[j]) for j in episode['supports']],
                'query':[int(ids[j]) for j in episode['query']], 'solved':episode['solved'],
                'used_updates':len(episode['supports']),'collection_s':episode['collection_s'],
                'loss':float(loss.detach()) if l1 is not None else None,
                'l1':float(l1) if l1 is not None else None,
                'geometry':float(geometry) if geometry is not None else None}


def preflight(team):
    prov=provenance();out=R/'preflight'/team.panel/team.kind
    model=team.model;initial=copy.deepcopy(model.state_dict());reader=old.reader_hash(model)
    probe=old.ident(team.probe);head=old.ident(team.head);records=[];verified_gradient=False
    opt,sched=old.optimizer(model)
    # Existing evaluation pool ensures the unchanged reader matches saved scores.
    task=team.plan['panels'][team.panel]['tasks'][0];team.load(task,True)
    scores=team.scores(model.initial_state(),True)
    baseline=old.baseline(team.panel,task,'q0',team.plan)['scores']
    equivalent_scores(scores,baseline)
    start=time.time();torch.cuda.reset_peak_memory_stats()
    for index in range(4):
        ep=team.collect(index,verify=True,cap_override=1 if index<2 else 8)
        if ep['supports'] and not verified_gradient:
            checks=[]
            for whole in [False,True]:
                model.zero_grad(set_to_none=True);model.checkpoint_blocks=not whole
                with controlled_backward('cpu',True):
                    loss,l1,geo=team.loss(ep,whole,True);loss.backward()
                checks.append({'loss':float(loss),'gradient':old.v.fingerprint(
                    (n,p.grad) for n,p in model.named_parameters() if p.grad is not None)})
                del loss
            assert checks[0]==checks[1],checks
            verified_gradient=True;model.checkpoint_blocks=False
        opt.zero_grad(set_to_none=True)
        loss,l1,geo=team.loss(ep,True,True)
        if loss.requires_grad:
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(norm) and norm>0;opt.step()
        sched.step();records.append(team.event(ep,loss,l1,geo));del loss,ep
    assert verified_gradient,'No nonempty episode for gradient validation'
    assert old.reader_hash(model)==reader and old.ident(team.probe)==probe and old.ident(team.head)==head
    if team.kind=='zero':
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=model.initial_state();res=torch.randn(1,32,196,128,device='cuda');a=torch.randn(1,3,device='cuda')
            assert torch.equal(model.update_state(state,res,a),model.update_state(state,res*3,a))
    model.load_state_dict(initial);model.zero_grad(set_to_none=True)
    data.write(out/'complete.json',{'passed':True,'provenance':prov,'records':records,
               'distributed_scores_exact':True,'same_initial_Q0_scores':True,'checkpoint_gradient_exact':True,
               'current_residual_replay_exact':True,'frozen_reader_probe_head':True,
               'world_size':team.world,'seconds':time.time()-start,'peak_GiB':torch.cuda.max_memory_allocated()/2**30})
    print('PREFLIGHT_PASS',team.panel,team.kind,flush=True)


def rankcheck(team):
    records=[]
    for task in team.plan['panels'][team.panel]['tasks'][:2]:
        team.request(mode='pool',task=task,canonical=True)
        initial=team.model.initial_state()
        for updated in [False,True]:
            state=initial if not updated else initial+.03*torch.randn_like(initial)
            scores=team.scores(state,True)
            if not updated:equivalent_scores(scores,old.baseline(team.panel,task,'q0',team.plan)['scores'])
            records.append({'task':task,'updated_state':updated,'exact':True})
    data.write(R/'preflight'/f'rankcheck_{team.panel}.json',
               {'passed':True,'world_size':team.world,'provenance':provenance(),'records':records})
    print('RANKCHECK_PASS',team.panel,team.world,flush=True)


def train(team):
    prov=provenance();folder=R/'models'/team.panel/team.kind;folder.mkdir(parents=True,exist_ok=True)
    assert data.read(R/'DATA_AUDIT.json')['provenance']==prov
    assert data.read(R/'preflight'/team.panel/team.kind/'complete.json')['provenance']==prov
    if (folder/'complete.json').exists():
        info=data.read(folder/'complete.json');assert info['provenance']==prov
        assert data.sha(info['checkpoint'])==info['checkpoint_sha256'];return
    model=team.model;reader=old.reader_hash(model);initial=old.ident(model)
    probe_hash=old.ident(team.probe);head_hash=old.ident(team.head)
    opt,sched=old.optimizer(model);start=updates=empty=supports=0;elapsed=0.;timings=[]
    resume=folder/'resume.pt';journal=folder/'episodes.jsonl';total=10000
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['provenance']==prov and ck['initial_sha256']==initial
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['slots'];updates=ck['optimizer_steps'];empty=ck['empty'];supports=ck['supports'];elapsed=ck['elapsed_s'];timings=ck['timings']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
    if journal.exists():
        keep=[]
        for line in journal.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:break
            if row['slot']<start:keep.append(line)
        assert len(keep)==start;journal.write_text(''.join(line+'\n' for line in keep))
    else:assert start==0
    begun=time.time();torch.cuda.reset_peak_memory_stats()
    with journal.open('a',buffering=1) as output:
        for slot in range(start,total):
            t0=time.time();opt.zero_grad(set_to_none=True);active=False;events=[]
            for offset in range(2):
                ep=team.collect(2*slot+offset)
                loss,l1,geo=team.loss(ep,True,verify=slot<3)
                assert torch.isfinite(loss)
                if loss.requires_grad:(loss/2).backward();active=True
                events.append(team.event(ep,loss,l1,geo));del loss,ep
            if active:
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                assert torch.isfinite(norm) and norm>0
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                opt.step()
            sched.step();updates+=int(active);empty+=sum(e['used_updates']==0 for e in events)
            supports+=sum(e['used_updates'] for e in events);seconds=time.time()-t0
            timings.append(seconds);timings=timings[-200:];done=slot+1
            output.write(json.dumps({'slot':slot,'optimizer_step':active,'events':events,'seconds':seconds})+'\n')
            if done%10==0 or done==total:
                progress={'slots':done,'total':total,'optimizer_steps':updates,'zero_support_episodes':empty,
                          'support_updates':supports,'elapsed_s':elapsed+time.time()-begun,
                          'recent_slot_mean_s':float(np.mean(timings)),
                          'projected_remaining_h':float(np.mean(timings))*(total-done)/3600,
                          'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'world_size':team.world,'provenance':prov}
                data.write(folder/'progress.json',progress)
                print('TRAIN',team.panel,team.kind,done,'mean_s',progress['recent_slot_mean_s'],
                      'remaining_h',progress['projected_remaining_h'],flush=True)
            if done%100==0 or done==total:
                assert old.reader_hash(model)==reader and old.ident(team.probe)==probe_hash and old.ident(team.head)==head_hash
                output.flush();os.fsync(output.fileno())
                old.b.atomic_save(resume,{'provenance':prov,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
                    'scheduler':sched.state_dict(),'slots':done,'optimizer_steps':updates,'empty':empty,'supports':supports,
                    'elapsed_s':elapsed+time.time()-begun,'initial_sha256':initial,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'timings':timings})
    checkpoint_path=folder/'model.pt'
    old.b.atomic_save(checkpoint_path,{'state_dict':model.state_dict(),'provenance':prov,'slots':total})
    data.write(folder/'complete.json',{'complete':True,'provenance':prov,'checkpoint':str(checkpoint_path),
        'checkpoint_sha256':data.sha(checkpoint_path),'initial_sha256':initial,'reader_sha256':reader,
        'probe_sha256':probe_hash,'head_sha256':head_hash,'slots':total,'optimizer_steps':updates,
        'episodes':2*total,'zero_support_episodes':empty,'support_updates':supports,
        'elapsed_s':elapsed+time.time()-begun,'episodes_sha256':data.sha(journal)})


class EvaluationObservations:
    def __init__(self):self.caches={}
    def get(self,panel,task,action_idx,action,z0,expected_status):
        previous=old.m.ASSETS/panel/'observations'/f'{task.replace(":","_")}_{action_idx:06d}.npz'
        if previous.exists():
            with np.load(previous) as z:
                assert str(z['task'])==task and int(z['status'])==int(expected_status)
                assert str(z['pca_sha256'])==data.sha(old.m.PCA)
                np.testing.assert_array_equal(z['action'],action);latent=z['latents'].copy()
            diff=np.abs(latent[0].astype(float)-z0)
            assert diff.max()<.01 and diff.mean()<1e-6
        else:
            if panel not in self.caches:self.caches[panel]=data.TruthCache(panel)
            pool={'action_idx':np.array([action_idx]),'action':action[None],
                  'z0':z0[None],'status':np.array([expected_status])}
            latent=self.caches[panel].get(task,0,pool)['latent']
        return torch.from_numpy(latent[None]).cuda().float()
    def close(self):
        for cache in self.caches.values():cache.close()


def evaluate(panel,kind,shard):
    # Keep the existing, audited evaluation implementation and baseline files.
    old.ROOT=R;old.provenance=provenance
    install_evaluation_guards()
    old.m.Observations=EvaluationObservations
    original_checked=old.plan_checked
    old.plan_checked=lambda:data.checked()
    # Old evaluator's panel gate uses our per-arm gate through a local wrapper.
    assert data.read(R/'preflight'/panel/kind/'complete.json')['provenance']==provenance()
    old_read=old.read
    def read(path):
        if Path(path)==R/'preflight'/panel/'complete.json':
            return {'provenance':provenance()}
        return old_read(path)
    old.read=read
    old.evaluate(panel,kind,shard)


def smoke_pools(panel,index=None):
    plan=data.checked();m=old.m
    encoder=data.EncoderProxy();base,head,rule,_=m.load_base(panel);worker=data.CPUWorker()
    try:
        for episode_index in (range(4) if index is None else [index]):
            task=draw_episode(plan['panels'][panel]['train_tasks'],episode_index)[0]
            data.build_pool(panel,task,encoder,base,head,rule,worker)
    finally:worker.close();encoder.close()


def summary():
    plan=data.checked();old.ROOT=R;old.provenance=provenance
    install_evaluation_guards()
    result={'complete':True,'provenance':provenance(),'panels':{},'training':{},
            'reused_controls':str(data.OLD/'SUMMARY.json')}
    previous=data.read(data.OLD/'SUMMARY.json')
    for panel,spec in plan['panels'].items():
        result['panels'][panel]={k:previous['panels'][panel][k] for k in ['base','q0','tta']}
        result['training'][panel]={}
        for kind in ['residual','zero']:
            complete=data.read(R/'models'/panel/kind/'complete.json')
            assert complete['provenance']==provenance() and complete['slots']==10000
            assert data.sha(complete['checkpoint'])==complete['checkpoint_sha256']
            assert data.sha(R/'models'/panel/kind/'episodes.jsonl')==complete['episodes_sha256']
            result['training'][panel][kind]=complete
            rows=[old.audit_case(R/'evaluation'/panel/kind/t.replace(':','_'),plan)[0] for t in spec['tasks']]
            first={r['task']:r['first_success'] for r in rows}
            arm={'n':len(rows),'first_success':first,'counts':{},'paired_Q0':{}}
            for k in [1,3,5,9]:
                solved=lambda n:n is not None and n<=k
                q0=previous['panels'][panel]['q0']['first_success']
                arm['counts'][f's{k}']=sum(solved(n) for n in first.values())
                arm['paired_Q0'][str(k)]={'gained':[t for t in first if solved(first[t]) and not solved(q0[t])],
                                          'lost':[t for t in first if not solved(first[t]) and solved(q0[t])]}
            result['panels'][panel][kind]=arm
    data.write(R/'SUMMARY.json',result)


def audit_data():
    plan=data.checked();panels={}
    for panel,spec in plan['panels'].items():
        positives=[];receipts={}
        for task in spec['train_tasks']:
            pool=data.load_pool(panel,task);folder=data.folder_for(panel,task)
            receipt=data.read(folder/'complete.json')
            assert data.sha(folder/'pool.npz')==receipt['pool_sha256']
            assert len(pool['action_idx'])==1024 and (pool['action_idx']>=0).all() and (pool['action_idx']<5000).all()
            assert np.all(pool['selection_score'][:-1]>=pool['selection_score'][1:])
            positives.append(int((pool['status']>0).sum()));receipts[task]=receipt
        panels[panel]={'scenes':len(positives),'candidates_per_scene':1024,
                       'mean_success_fraction':float(np.mean(positives)/1024),
                       'median_success_fraction':float(np.median(positives)/1024),
                       'scenes_with_success':sum(n>0 for n in positives),'receipts':receipts}
    data.write(R/'DATA_AUDIT.json',{'passed':True,'provenance':provenance(),'panels':panels})
    print('DATA_AUDIT_PASS',{p:{k:v for k,v in x.items() if k!='receipts'} for p,x in panels.items()},flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','train','eval','summary','smoke_pools','pool_one','rankcheck','audit_data'])
    ap.add_argument('--panel',choices=['cross','within']);ap.add_argument('--kind',choices=['residual','zero'])
    ap.add_argument('--shard',type=int,default=0);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);random.seed(0)
    if args.mode=='audit_data':audit_data();return
    if args.mode=='pool_one':smoke_pools(args.panel,args.shard);return
    if args.mode=='smoke_pools':smoke_pools(args.panel);return
    if args.mode=='eval':evaluate(args.panel,args.kind,args.shard);return
    if args.mode=='summary':summary();return
    rank=int(os.environ.get('SLURM_PROCID','0'));world=int(os.environ.get('SLURM_NTASKS','1'))
    assert torch.cuda.device_count()==1,os.environ.get('CUDA_VISIBLE_DEVICES')
    torch.cuda.set_device(0)
    dist.init_process_group('gloo',rank=rank,world_size=world,timeout=timedelta(minutes=45))
    team=Team(args.panel,args.kind,data.checked())
    try:
        if rank:
            team.serve()
        else:
            if args.mode=='rankcheck':rankcheck(team)
            elif args.mode=='preflight':preflight(team)
            else:train(team)
            team.request(mode='stop')
    finally:
        team.truth.close();dist.destroy_process_group()


if __name__=='__main__':main()
