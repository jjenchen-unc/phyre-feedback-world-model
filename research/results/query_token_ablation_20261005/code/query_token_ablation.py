"""Five-arm, paired-data query-token experiment; fixed reference, sampled prefix.

Independent run: does not modify historical predictor/memory experiments.
"""
import argparse
import copy
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import fresh_memory_geometry as f
from common import Predictor
from query_token_predictor import QueryTokenPredictor

ARMS = ['A_frozen_memory', 'B_pretrained_memory', 'C_random_memory',
        'D_pretrained_plain', 'E_random_plain']
PARENT = f.PROJECT / 'results/memory_fresh_geometry_20261005'


class Model(QueryTokenPredictor):
    def __init__(self, backbone, reference, din, arm, **architecture):
        super().__init__(backbone, din, **architecture)
        # Reference is externally owned, frozen, and never an optimizer parameter.
        object.__setattr__(self, 'reference', reference)
        self.arm = arm
        self.use_memory = arm in ARMS[:3]
        if not self.use_memory:
            self.requires_grad_(False)
        self.base.requires_grad_(arm != ARMS[0])

    @torch.no_grad()
    def context_residual(self, ctx_full, ctx_act):
        b, k = ctx_act.shape[:2]
        return torch.stack([torch.stack([
            (ctx_full[j, i:i+1] - self.reference(ctx_full[j, i:i+1, 0].contiguous(),
                                              ctx_act[j, i:i+1]))[0]
            for i in range(k)]) for j in range(b)])

    def forward(self, z0, action, memory=None, mask=None):
        if not self.use_memory:
            return self.base(z0, action)
        return super().forward(z0, action, memory, mask)


def pooled(residual, steps=16, grid=10):
    b, t, s, d = residual.shape
    side = int(round(s**.5))
    assert side * side == s and t % steps == 0
    z = residual[:, ::t//steps][:, :steps]
    z = z.reshape(b*steps, side, side, d).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(z, grid).permute(0, 2, 3, 1).reshape(b, steps, grid*grid, d)


def models(m, arm, seed, checkpoint=None):
    ref, probe, rule, pc, _ = f.load_models(memory=None)
    ref.eval().requires_grad_(False)
    probe.eval().requires_grad_(False)
    torch.manual_seed(seed)
    if arm in [ARMS[2], ARMS[4]]:
        a = pc['args']
        backbone = Predictor(pc['Din'], pc['T'], pc['S'], a['d'], a['layers'], a['heads']).cuda()
    else:
        backbone = copy.deepcopy(ref)
    # All memory arms use the same token initialization for a given seed.
    torch.manual_seed(seed + 7919)
    model = Model(backbone, ref, pc['Din'], arm, **m['architecture']).cuda()
    if checkpoint is not None:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert ck['arm'] == arm and ck['seed'] == seed
        assert ck['provenance'] == f.provenance(Path(m['run_root']))
        model.load_state_dict(ck['state_dict'], strict=True)
    return ref, probe, rule, model


def episode(rows, seed, index):
    row, ids, times, _ = f.episode(rows, seed, index)
    rng = np.random.default_rng([f.SEED, seed, index, 9])
    requested = int(rng.choice([1, 2, 4, 8]))
    ctx = rng.choice(row['failures'], min(requested, len(row['failures'])), replace=False).tolist()
    length = int(np.random.default_rng([f.SEED, seed, index, 10]).integers(1, len(ctx)+1))
    return {'task': row['task'], 'query': ids, 'times': times, 'support': ctx,
            'requested': requested, 'prefix': length, 'used_support': ctx[:length]}


def prepare(root):
    assert not (root/'manifest.json').exists()
    m = f.read(PARENT/'manifest.json')
    audit = f.read(PARENT/'DATA_AUDIT.json')
    assert audit['passed'] and audit['manifest_sha256'] == f.digest(PARENT/'manifest.json')
    m['parent_manifest_sha256'] = f.digest(PARENT/'manifest.json')
    m['run_root'] = str(root)
    m['scope'] = 'Five-arm fixed-budget supervised experiment; fresh modules, pretrained or random predictor; no policy replay training yet.'
    m['initialization'] = 'A frozen pretrained; B/D trainable pretrained; C/E random predictor; all token modules fresh.'
    m['arms'] = ARMS
    m['architecture'] = {'n_tokens':32, 'd_mem':256, 'mem_steps':16, 'mem_grid':10}
    m['training'].update({'episodes':20000, 'scene_batch':2, 'steps':10000,
        'support_lengths':[1,2,4,8], 'prefix_loss':'Uniform one prefix per episode; full gradient through preceding updates; equal scene weight',
        'batch_policy':'Two scenes per optimizer update, <=8 queries each; identical draws for all arms',
        'geometry_weight':.1})
    m['checkpoints'] = [0,500,2000,5000,10000,20000] # episode counts, not optimizer updates
    m['cache_root'] = '/mnt/meg/jjenchen/phyre-unc/runs/query_token_ablation_20261005/residual_cache'
    m['cache_shards'] = 4
    m['residual'] = 'truth minus immutable pretrained reference; bf16 batch1 prediction, fp32 residual and fixed pooling16x10x10; cache fp32'
    m['evaluation'] = {'heads':['original',m['primary_head']], 'closed_budget':20,
        'cross_per_template':5, 'within_per_template':2,
        'prefixes':[1,2,4,8], 'ablations':['full','latest','zero','static'],
        'primary_in_range':'S@9; S@20 includes extrapolation beyond8updates',
        'selection':'final checkpoint only; curves on calibration only'}
    m['limits'] = ['Fixed20Kepisodes is not proof of convergence, especially random initialization',
        'Same LR and scheduler across arms; not a per-arm optimal-hyperparameter comparison',
        'Frozen encoder, probe and residual reference remain pretrained even in random predictor arms',
        'Support training is random failure replay, NOT on-policy; policy collection is a later stage',
        'Reused cross dev; within fresh generated seen-template tasks, not official within fold',
        'Same data, optimizer steps and losses, but different FLOPs and parameter counts',
        'Closed-loop paths differ across arms; fixed-prefix diagnostics isolate shared histories',
        'Evaluation-only ablations can be out of distribution; no unique causal proof',
        'Closed-loop20tasks/split is exploratory; report per-task and seeds']
    m['code'] = f.code_hashes()
    f.write_json(root/'manifest.json',m)
    store = f.Store(include_features=False)
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest() == m['store_files_sha256']
    for p,h in m['source_hashes'].items(): assert f.digest(p)==h
    rows = [r for r in m['rows'] if r['split']=='train']
    draws = {}
    for seed in m['seeds']:
        h=hashlib.sha256(); scenes=set(); depths=Counter()
        for i in range(m['training']['episodes']):
            ep=episode(rows,seed,i); q=ep['query']; ctx=ep['used_support']
            assert not set(q)&set(ctx) and len(ctx)==len(set(ctx)) and not store.labels[ctx].any()
            assert all(str(store.tasks[j])==ep['task'] for j in q+ctx)
            h.update(json.dumps(ep,sort_keys=True).encode());scenes.add(ep['task']);depths[ep['prefix']]+=1
        assert len(scenes)==1200
        draws[str(seed)]={'sha256':h.hexdigest(),'scenes':len(scenes),'prefix_counts':dict(depths)}
    f.write_json(root/'DATA_AUDIT.json',{'passed':True,'provenance':f.provenance(root),
        'parent_audit_sha256':f.digest(PARENT/'DATA_AUDIT.json'),'near_filter':False,'draws':draws,'counts':m['counts']})
    print('PREPARED',json.dumps(draws),flush=True)


def checked(root):
    m=f.read(root/'manifest.json')
    assert m['code']==f.code_hashes()
    assert f.digest(m['base'])==m['base_sha256'] and f.digest(m['probe'])==m['probe_sha256']
    assert f.read(root/'DATA_AUDIT.json')['provenance']==f.provenance(root)
    return m


def cache(root, shard):
    m=checked(root); folder=Path(m['cache_root']);folder.mkdir(parents=True,exist_ok=True)
    ids=sorted({i for r in m['rows'] for i in r['failures']})[shard::m['cache_shards']]
    out=folder/f'shard_{shard}.npy'; done=folder/f'shard_{shard}.json'
    if done.exists():
        d=f.read(done);assert d['provenance']==f.provenance(root) and d['sha256']==f.digest(out);return
    store=f.Store(include_features=False);ref,_,_,_,_=f.load_models(memory=None);ref.eval().requires_grad_(False)
    z=store.z(ids[:1]);shape=(len(ids),16,100,z.shape[-1])
    tmp=out.with_suffix('.tmp.npy'); arr=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=shape)
    start=time.time()
    with torch.no_grad():
        for j,i in enumerate(ids):
            z=torch.from_numpy(store.z([i])).cuda().float();a=torch.from_numpy(store.actions[[i]]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16): pr=ref(z[:,0].contiguous(),a)
            arr[j]=pooled(z-pr.float())[0].cpu().numpy()
            if (j+1)%1000==0:print('CACHE',shard,j+1,len(ids),round(time.time()-start),flush=True)
    arr.flush();del arr;tmp.replace(out)
    f.write_json(done,{'provenance':f.provenance(root),'ids':ids,'shape':shape,'sha256':f.digest(out),'seconds':time.time()-start})


class Cache:
    def __init__(self,m):
        self.arr=[];self.lookup={}
        for s in range(m['cache_shards']):
            p=Path(m['cache_root'])/f'shard_{s}.json';d=f.read(p)
            assert d['provenance']==f.provenance(Path(m['run_root']))
            self.arr.append(np.load(p.with_suffix('.npy'),mmap_mode='r'))
            for j,i in enumerate(d['ids']):assert i not in self.lookup;self.lookup[i]=(s,j)
    def get(self,ids):
        return np.stack([self.arr[self.lookup[i][0]][self.lookup[i][1]] for i in ids])


def states(model, episodes, store, cache_store):
    state=model.initial_state(len(episodes))
    if not model.use_memory:return state
    for j in range(max(e['prefix'] for e in episodes)):
        alive=[i for i,e in enumerate(episodes) if j<e['prefix']]
        ii=torch.tensor(alive,device=state.device)
        ids=[episodes[i]['used_support'][j] for i in alive]
        r=torch.from_numpy(cache_store.get(ids)).cuda()
        a=torch.from_numpy(store.actions[ids]).cuda()
        updated=model.update_state(state.index_select(0,ii),r,a)
        state=state.index_copy(0,ii,updated)
    return state


def batch_loss(model,probe,episodes,store,cache_store):
    ids=[j for e in episodes for j in e['query']]
    owners=[i for i,e in enumerate(episodes) for _ in e['query']]
    z=torch.from_numpy(store.z(ids)).cuda().float();a=torch.from_numpy(store.actions[ids]).cuda()
    features=[store.feature(i) for i in ids]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        state=states(model,episodes,store,cache_store)
        memory=state[torch.tensor(owners,device=state.device)]
        mask=torch.ones(len(ids),model.n_tokens,dtype=torch.bool,device=state.device)
        pred=model(z[:,0].contiguous(),a,memory,mask).float()
    assert torch.isfinite(state).all() and torch.isfinite(pred).all()
    ls=[];gs=[];offset=0
    for e in episodes:
        n=len(e['query']);sl=slice(offset,offset+n)
        ls.append((pred[sl,1:]-z[sl,1:]).abs().mean())
        gs.append(f.geometry_loss(probe,pred[sl],features[sl],e['times']))
        offset+=n
    l=torch.stack(ls).mean();g=torch.stack(gs).mean()
    return l+.1*g,l,g


def save_model(path,model,root,arm,seed,episodes):
    f.atomic_torch(path,{'arm':arm,'seed':seed,'episodes':episodes,'provenance':f.provenance(root),
                        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}})


@torch.no_grad()
def observed_state(model,source,ids,variant='full'):
    state=model.initial_state()
    if not model.use_memory or not ids or variant=='static':return state
    use=ids[-1:] if variant=='latest' else ids
    for i in use:
        z=torch.from_numpy(source.z([i])).cuda().float();a=torch.from_numpy(source.actions[[i]]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            r=model.context_residual(z[:,None],a[:,None])[:,0].float()
            if variant=='zero':r=torch.zeros_like(r)
            state=model.update_state(state,r,a)
    return state


@torch.no_grad()
def panel(folder,spec,source,ref,model,heads,rule,prov,variant='full'):
    if (folder/'summary.json').exists():
        r=f.read(folder/'summary.json');assert r['provenance']==prov;return r
    folder.mkdir(parents=True,exist_ok=True)
    ids,ctx=spec['indices'],spec['context'];query=spec['query']
    assert not set(query)&set(ctx) and not source.labels[ctx].any()
    assert all(str(source.tasks[i])==spec['task'] for i in ids+ctx)
    model.eval();state=observed_state(model,source,ctx,variant)
    features=[source.feature(i) for i in ids];labels=source.labels[ids];qset=set(query)
    truth={'indices':np.asarray(ids),'labels':labels,'query_mask':np.array([i in qset for i in ids]),
           'distance':np.stack([f.surface_dist(v) for v in features]),'xy':np.stack([v[:,:,:2]*256 for v in features])}
    collected=defaultdict(lambda:defaultdict(list))
    for start in range(0,len(ids),32):
        ii=ids[start:start+32];z=torch.from_numpy(source.z(ii)).cuda().float();a=torch.from_numpy(source.actions[ii]).cuda()
        memory,mask=model.state_memory(state,bool(ctx))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            bp=ref(z[:,0].contiguous(),a).float()
            mp=model(z[:,0].contiguous(),a,memory.expand(len(ii),-1,-1),mask.expand(len(ii),-1)).float()
            if start==0:
                empty,em=model.state_memory(model.initial_state(),False)
                ep=model(z[:,0].contiguous(),a,empty.expand(len(ii),-1,-1),em.expand(len(ii),-1)).float()
                own=model.base(z[:,0].contiguous(),a).float()
                assert torch.equal(ep,own) # trained backbone, not immutable reference!
        assert torch.equal(mp[:,0],z[:,0])
        for kind,pred in [('base',bp),('memory',mp)]:
            diff=pred[:,1:]-z[:,1:]
            for name,head in heads.items():
                v=f.readout(head,pred,features[start:start+32],rule)
                if start==0 and kind=='memory':
                    altered=[x.copy() for x in features[:len(ii)]]
                    for x in altered:x[1:]=123
                    same=f.readout(head,pred,altered,rule)
                    for k in v:np.testing.assert_array_equal(v[k],same[k])
                v.update(latent_l1=diff.abs().mean((1,2,3)).cpu().numpy(),latent_l2_rms=diff.square().mean((1,2,3)).sqrt().cpu().numpy())
                for k,value in v.items():collected[name+'__'+kind][k].append(value)
    raw={k:{field:np.concatenate(values) for field,values in dd.items()} for k,dd in collected.items()}
    green=int(np.flatnonzero(features[0][0,:,9]>.5)[0])
    goals=[o for o in range(features[0].shape[1]) if features[0][0,o,10]>.5 and features[0][0,o,13]<.5 and features[0][0,o,11]<.5]
    metrics={}
    for k,v in raw.items():
        h,kind=k.split('__');f.save_npz(folder/(k+'.npz'),v)
        metrics[k]=f.describe(v,truth,labels,truth['query_mask'],rule,green,goals,base=raw[h+'__base'] if kind=='memory' else None)
    f.save_npz(folder/'truth.npz',truth)
    result={**spec,'variant':variant,'rule':rule,'green':green,'dynamic_goals':goals,'conditions':metrics,
            'provenance':prov,'empty_identity':True,'empty_reference':'current arm backbone',
            'future_input_isolation':True}
    f.write_json(folder/'summary.json',result);f.audit_panel(folder)
    return result


def curve(root,folder,m,model,ref,probe,rule,store,episodes):
    specs=f.choose_tasks([r for r in m['curve_panels'] if r['panel']=='calibration'],1,'ablation_curve')
    out=[]
    for spec in specs:
        ids=spec['indices'][:16];spec=dict(spec,indices=ids,query=ids)
        p=folder/'curves'/str(episodes)/spec['task'].replace(':','_')
        out.append(panel(p,spec,store,ref,model,{'original':probe},rule,{**f.provenance(root),'episodes':episodes}))
    f.write_json(folder/'curves'/str(episodes)/'SUMMARY.json',f.aggregate(out,'original'))


@torch.no_grad()
def engineering(model,ref,probe,store,cache_store,episodes,root,arm,seed,folder):
    model.eval();e=next(e for e in episodes if len(e['support'])>=2)
    ids=e['support'][:2];z=torch.from_numpy(store.z(ids)).cuda().float();a=torch.from_numpy(store.actions[ids]).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        res=model.context_residual(z[None],a[None])[0].float()
    cached=torch.from_numpy(cache_store.get(ids)).cuda()
    assert torch.equal(pooled(res),cached)
    before=f.frozen_fingerprint(model)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        state=model.initial_state()
        for j in range(2):state=model.update_state(state,res[j:j+1],a[j:j+1])
        replay=observed_state(model,store,ids) if model.use_memory else state
        assert torch.equal(state,replay)
        memory,mask=model.state_memory(state,True)
        pred=model(z[:,0].contiguous(),a,memory.expand(2,-1,-1),mask.expand(2,-1))
        empty,em=model.state_memory(model.initial_state(),False)
        own=model.base(z[:,0].contiguous(),a)
        assert torch.equal(own,model(z[:,0].contiguous(),a,empty.expand(2,-1,-1),em.expand(2,-1)))
        packed,pm=model.encode_memory(z[None],a[None],torch.zeros(1,2,device='cuda'),torch.ones(1,2,device='cuda',dtype=torch.bool),res[None])
        assert torch.equal(packed,state)
        shifted,_=model.encode_memory(z[None]+100,a[None],torch.ones(1,2,device='cuda'),torch.ones(1,2,device='cuda',dtype=torch.bool),res[None])
        assert torch.equal(shifted,state)
    f.atomic_torch(folder/'state_roundtrip.pt',{'state':state.cpu()})
    saved=torch.load(folder/'state_roundtrip.pt',weights_only=False)['state'].cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        assert torch.equal(pred,model(z[:,0].contiguous(),a,saved.expand(2,-1,-1),mask.expand(2,-1)))
    assert before==f.frozen_fingerprint(model)
    save_model(folder/'roundtrip.pt',model,root,arm,seed,len(episodes))
    ck=torch.load(folder/'roundtrip.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state_dict'],strict=True)
    assert before==f.frozen_fingerprint(model)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        assert torch.equal(pred,model(z[:,0].contiguous(),a,saved.expand(2,-1,-1),mask.expand(2,-1)))
    return {'cache_exact':True,'two_update_replay':True,'empty_current_backbone_identity':True,
            'future_input_isolation':True,'state_and_full_checkpoint_roundtrip':True,'online_weights_unchanged':True}


def train(root,arm,seed,smoke=False):
    m=checked(root);prov=f.provenance(root);cfg=m['training']
    if not smoke:
        assert f.read(root/'SMOKE.json')['passed'] and f.read(root/'SMOKE.json')['provenance']==prov
    folder=root/('smoke' if smoke else 'models')/f'{arm}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        d=f.read(folder/'complete.json');assert d['provenance']==prov and f.digest(d['checkpoint'])==d['checkpoint_sha256'];return
    store=f.Store();cache_store=Cache(m);ref,probe,rule,model=models(m,arm,seed)
    rows=[r for r in m['rows'] if r['split']=='train']
    total=200 if smoke else cfg['episodes'];batch=cfg['scene_batch'];steps=total//batch
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,cfg['lr'],total_steps=steps,pct_start=.1 if smoke else .03)
    frozen={k:f.frozen_fingerprint(v) for k,v in [('reference',ref),('probe',probe)]}
    initial_backbone=f.frozen_fingerprint(model.base)
    initial_tokens=f.frozen_fingerprint(model)
    start=0;logs=[];draws=[];grads={};resume=folder/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['provenance']==prov and ck['total_episodes']==total
        model.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['episodes'];logs=ck['logs'];draws=ck['draws'];grads=ck['gradients']
        torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    if start==0 and not smoke:
        save_model(folder/'checkpoints/episode_000000.pt',model,root,arm,seed,0)
        curve(root,folder,m,model,ref,probe,rule,store,0)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.time();running=[]
    for offset in range(start,total,batch):
        eps=[episode(rows,seed,i) for i in range(offset,offset+batch)]
        model.train();ref.eval();probe.eval()
        if arm==ARMS[0]:model.base.eval()
        opt.zero_grad(set_to_none=True)
        loss,l,g=batch_loss(model,probe,eps,store,cache_store)
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['clip']);assert torch.isfinite(norm) and norm>0
        if offset in [0,batch]:
            bgrad=sum(float(p.grad.abs().sum()) for p in model.base.parameters() if p.grad is not None)
            assert (bgrad==0) == (arm==ARMS[0])
            grads['backbone']=bgrad
            if model.use_memory and offset==batch:
                for name,p in [('initial_tokens',model.initial_tokens),('update_attention',model.update_attention.in_proj_weight),('residual_in',model.residual_in.weight)]:
                    val=float(p.grad.abs().sum());assert val>0;grads[name]=val
        assert all(p.grad is None for p in ref.parameters()) and all(p.grad is None for p in probe.parameters())
        opt.step();sched.step();draws.extend(eps)
        running.append([float(l.detach()),float(g.detach()),float(loss.detach())])
        seen=offset+batch
        if seen%100==0 or seen==total:
            torch.cuda.synchronize()
            row={'episodes':seen,'optimizer_steps':seen//batch,'losses':np.mean(running,0).tolist(),
                 'seconds_this_attempt':time.time()-t0,'start_episode':start,
                 'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,'lr':sched.get_last_lr()[0]}
            logs.append(row);running=[];f.write_json(folder/'progress.json',row);print('TRAIN',arm,seed,json.dumps(row),flush=True)
        if not smoke and seen in m['checkpoints']:
            save_model(folder/f'checkpoints/episode_{seen:06d}.pt',model,root,arm,seed,seen)
            curve(root,folder,m,model,ref,probe,rule,store,seen)
        if seen%500==0 or seen==total:
            f.atomic_torch(resume,{'provenance':prov,'total_episodes':total,'episodes':seen,
                'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'logs':logs,'draws':draws,'gradients':grads})
    assert frozen=={k:f.frozen_fingerprint(v) for k,v in [('reference',ref),('probe',probe)]}
    assert (f.frozen_fingerprint(model.base)==initial_backbone)==(arm==ARMS[0])
    assert f.frozen_fingerprint(model)!=initial_tokens
    checks=engineering(model,ref,probe,store,cache_store,draws,root,arm,seed,folder) if smoke else {}
    if smoke:
        assert {1,2,4,8} <= {e['prefix'] for e in draws}
        spec=next(r for r in m['curve_panels'] if r['panel']=='calibration')
        spec=dict(spec,indices=spec['indices'][:16],query=spec['indices'][:16])
        panel(folder/'panel_smoke',spec,store,ref,model,{'original':probe},rule,prov)
        source=f.Pack(m['cross_pack'])
        spec=next(r for r in m['final_panels'] if r['panel']=='cross_natural' and r['context'] and source.labels[r['indices']].any())
        heads,_=f.load_heads(m,f.PROBES,probe)
        closed_case(folder/'closed_smoke',spec,source,model,ref,heads[m['primary_head']],rule,prov,budget=3)
        with np.load(folder/'closed_smoke/snapshots.npz') as p:
            with np.load(f.PROBES/'evaluation/cross'/spec['task'].replace(':','_')/(m['primary_head']+'__base.npz')) as old:
                np.testing.assert_array_equal(p['base_scores'],old['log_score'])
        checks['fixed_metrics_and_closed_loop_smoke']=True
        checks['historic_reference_scores_exact']=True
        checks['gpu_depths_1_2_4_8']=True
    ck=folder/f'checkpoints/episode_{total:06d}.pt';save_model(ck,model,root,arm,seed,total)
    h=hashlib.sha256()
    for e in draws:h.update(json.dumps(e,sort_keys=True).encode())
    if not smoke:assert h.hexdigest()==f.read(root/'DATA_AUDIT.json')['draws'][str(seed)]['sha256']
    f.write_json(folder/'draws.json',draws)
    f.write_json(folder/'complete.json',{'complete':True,'provenance':prov,'arm':arm,'seed':seed,'episodes':total,
        'optimizer_steps':steps,'draw_sha256':h.hexdigest(),'initial_backbone_sha256':initial_backbone,
        'final_backbone_sha256':f.frozen_fingerprint(model.base),'frozen_reference_probe':True,
        'checkpoint':str(ck),'checkpoint_sha256':f.digest(ck),'gradients':grads,'engineering':checks,'logs':logs})


def smoke_summary(root):
    m=checked(root);rows=[]
    for arm in ARMS:
        d=f.read(root/'smoke'/f'{arm}_s0/complete.json');assert d['complete'] and d['provenance']==f.provenance(root)
        assert all(d['engineering'].values());rows.append(d)
    assert len({d['draw_sha256'] for d in rows})==1
    assert rows[1]['initial_backbone_sha256']==rows[3]['initial_backbone_sha256']==rows[0]['initial_backbone_sha256']
    assert rows[2]['initial_backbone_sha256']==rows[4]['initial_backbone_sha256']!=rows[0]['initial_backbone_sha256']
    f.write_json(root/'SMOKE.json',{'passed':True,'provenance':f.provenance(root),
        'same_draws':True,'paired_initialization':True,'arms':{d['arm']:{'logs':d['logs'],'gradients':d['gradients'],'engineering':d['engineering']} for d in rows}})
    print('FIVE_ARM_SMOKE_PASS',flush=True)


@torch.no_grad()
def scores(model,state,observed,z0,actions,initial_features,head,rule):
    """Candidate scoring cannot access true future features, outcomes or latents."""
    values=[]
    for start in range(0,len(actions),32):
        sl=slice(start,start+32)
        z=torch.from_numpy(z0[sl].copy()).cuda().float();a=torch.from_numpy(actions[sl]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if isinstance(model,Model):
                mem,mask=model.state_memory(state,observed)
                pr=model(z,a,mem.expand(len(z),-1,-1),mask.expand(len(z),-1)).float()
            else:pr=model(z,a).float()
        values.append(f.readout(head,pr,initial_features[sl],rule)['log_score'])
    result=np.concatenate(values);assert np.isfinite(result).all();return result


def area20(first):
    w=np.diff(np.log(np.arange(1,22)))
    return float(w[np.arange(1,21)>=first].sum()/w.sum()) if first else 0.


@torch.no_grad()
def closed_case(folder,spec,source,model,ref,head,rule,prov,variant='full',budget=20):
    if (folder/'trace.json').exists():
        d=f.read(folder/'trace.json');assert d['provenance']==prov;return d
    folder.mkdir(parents=True,exist_ok=True);model.eval();ref.eval()
    ids=spec['indices'];assert len(ids)==1024
    # Future latents/outcomes are not available to scores().
    z0=np.array(source.lat[0][ids,0])
    actions=source.actions[ids];initial=[source.feature(i)[:1].copy() for i in ids]
    baseline=scores(ref,None,False,z0,actions,initial,head,rule)
    state=model.initial_state().detach().clone();seen=[];attempts=[];snapshots=[];states_saved=[state.cpu().numpy().copy()]
    weight_before=f.frozen_fingerprint(model);first=0;tried=np.zeros(len(ids),bool)
    current=scores(model,state,False,z0,actions,initial,head,rule)
    if model.arm==ARMS[0]:np.testing.assert_array_equal(current,baseline)
    for attempt in range(1,budget+1):
        snapshots.append(current.copy());local=int(np.argmax(np.where(tried,-np.inf,current)))
        assert not tried[local];tried[local]=True;idx=ids[local]
        solved=bool(source.labels[idx])
        attempts.append({'attempt':attempt,'local':local,'index':idx,'solved':solved,'score':float(current[local])})
        if solved:first=attempt;break
        if attempt==budget:break
        seen.append(idx)
        if model.use_memory:
            # Only the selected, observed failure enters the state update.
            z=torch.from_numpy(source.z([idx])).cuda().float();a=torch.from_numpy(actions[local:local+1]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                residual=model.context_residual(z[:,None],a[:,None])[:,0].float()
                if variant=='latest':state=model.initial_state()
                state=model.update_state(state,residual,a)
            # Independent replay from Q0 confirms causal residual -> state chain.
            replay=observed_state(model,source,seen,variant)
            assert torch.equal(state,replay)
        states_saved.append(state.cpu().numpy().copy())
        current=scores(model,state,True,z0,actions,initial,head,rule)
    assert weight_before==f.frozen_fingerprint(model)
    # CPU round-trip of the final state must reproduce the actual final ranking.
    restored=torch.from_numpy(states_saved[-1]).cuda()
    reproduced=scores(model,restored,len(seen)>0,z0,actions,initial,head,rule)
    np.testing.assert_array_equal(reproduced,snapshots[-1])
    y=source.labels[ids];base_order=np.argsort(-baseline,kind='stable')[:budget]
    bf=int(np.flatnonzero(y[base_order])[0])+1 if y[base_order].any() else 0
    result={'task':spec['task'],'first_success':first,'base_first_success':bf,
            'success9':bool(0<first<=9),'success20':bool(first),'base_success9':bool(0<bf<=9),
            'base_success20':bool(bf),'log_area20':area20(first),'base_log_area20':area20(bf),'updates':len(seen)}
    f.save_npz(folder/'snapshots.npz',{'indices':np.asarray(ids),'labels':y,'base_scores':baseline,
        'scores':np.stack(snapshots),'tokens':np.concatenate(states_saved)})
    d={'provenance':prov,'variant':variant,'budget':budget,'attempts':attempts,'result':result,
       'state_replay_exact':True,'restored_state_score_replay_exact':True,'network_weights_unchanged':True}
    f.write_json(folder/'trace.json',d);audit_closed(folder);return d


def audit_closed(folder):
    d=f.read(folder/'trace.json')
    with np.load(folder/'snapshots.npz') as p:
        ids=p['indices'];y=p['labels'];ss=p['scores'];tokens=p['tokens'];base=p['base_scores']
    budget=d['budget'];assert len(ss)==len(d['attempts'])==len(tokens) and len(ss)<=budget
    tried=np.zeros(len(ids),bool);first=0
    for j,a in enumerate(d['attempts']):
        local=int(np.argmax(np.where(tried,-np.inf,ss[j])))
        assert local==a['local'] and int(ids[local])==a['index'] and float(ss[j,local])==a['score']
        assert a['solved']==bool(y[local]);tried[local]=True
        if a['solved']:assert j==len(ss)-1;first=j+1
    assert first or len(ss)==budget
    order=np.argsort(-base,kind='stable')[:budget]
    bf=int(np.flatnonzero(y[order])[0])+1 if y[order].any() else 0
    r=d['result'];assert first==r['first_success'] and bf==r['base_first_success']
    assert r['success9']==bool(0<first<=9) and r['success20']==bool(first)
    assert r['log_area20']==area20(first) and r['base_log_area20']==area20(bf)
    assert all(d[k] for k in ['state_replay_exact','restored_state_score_replay_exact','network_weights_unchanged'])
    return d


def evaluate(root,arm,seed,within):
    m=checked(root);folder=root/('within_evaluation' if within else 'evaluation')/f'{arm}_s{seed}'
    done=f.read(root/'models'/f'{arm}_s{seed}/complete.json')
    assert done['complete'] and done['episodes']==m['training']['episodes'] and done['provenance']==f.provenance(root)
    ck=Path(done['checkpoint']);assert f.digest(ck)==done['checkpoint_sha256']
    prov={**f.provenance(root),'checkpoint':f.digest(ck),'arm':arm,'seed':seed}
    if (folder/'complete.json').exists():assert f.read(folder/'complete.json')['provenance']==prov;return
    ref,probe,rule,model=models(m,arm,seed,ck);model.eval()
    allheads,hh=f.load_heads(m,f.PROBES,probe)
    assert all(hh[h]==v for h,v in m['heads'].items())
    heads={h:allheads[h] for h in m['evaluation']['heads']}
    if within:
        wm=f.read(root/'within_inputs.json');assert wm['provenance']==f.provenance(root)
        assert f.digest(wm['pack']+'_meta.npz')==wm['meta_sha256']
        source=f.Pack(wm['pack']);specs=wm['specs'];per=m['evaluation']['within_per_template']
    else:
        assert f.digest(m['cross_pack']+'_meta.npz')==m['cross_meta_sha256']
        source=f.Pack(m['cross_pack']);specs=[r for r in m['final_panels'] if r['panel']=='cross_natural'];per=m['evaluation']['cross_per_template']
    rows=[];diagnostics=[];closed=[];t0=time.time()
    for spec in specs:
        rows.append(panel(folder/'fixed'/spec['task'].replace(':','_'),spec,source,ref,model,heads,rule,prov))
        print('EVAL_FIXED',arm,seed,spec['task'],round(time.time()-t0),flush=True)
    selected=f.choose_tasks(specs,per,'ablation_closed')
    for spec in selected:
        name=spec['task'].replace(':','_')
        for variant in (['full','latest'] if model.use_memory else ['full']):
            closed.append(closed_case(folder/'closed'/variant/name,spec,source,model,ref,heads[m['primary_head']],rule,prov,variant))
        # All arms see the same fixed-reference failed prefix, stopped at success.
        with np.load(folder/'closed/full'/name/'snapshots.npz') as p:base_scores=p['base_scores']
        ids=spec['indices'];ctx=[]
        for local in np.argsort(-base_scores,kind='stable'):
            idx=ids[int(local)]
            if source.labels[idx] or len(ctx)==8:break
            ctx.append(idx)
        if ctx:
            query=sorted(sorted([i for i in ids if i not in ctx],key=lambda i:f.stable(('ablation_diag',spec['task'],i)))[:64])
            for k in sorted({min(k,len(ctx)) for k in m['evaluation']['prefixes']}):
                ss=dict(spec,context=ctx[:k],indices=query,query=query)
                for variant in (m['evaluation']['ablations'] if model.use_memory else ['full']):
                    path=folder/'diagnostics'/name/str(k)/variant
                    diagnostics.append(panel(path,ss,source,ref,model,{m['primary_head']:heads[m['primary_head']]},rule,prov,variant))
        print('EVAL_CLOSED_DIAG',arm,seed,spec['task'],round(time.time()-t0),flush=True)
    out={h:f.aggregate(rows,h) for h in heads}
    f.write_json(folder/'complete.json',{'complete':True,'provenance':prov,'fixed_scenes':len(rows),
        'closed_cases':len(closed),'diagnostic_cases':len(diagnostics),'metrics':out,'elapsed_seconds':time.time()-t0})


def summarize(root,within):
    m=checked(root);base=root/('within_evaluation' if within else 'evaluation');out={};paired={};arrays=0
    fixed={};closed={}
    for seed in m['seeds']:
        for arm in ARMS:
            folder=base/f'{arm}_s{seed}';d=f.read(folder/'complete.json');train_done=f.read(root/'models'/f'{arm}_s{seed}/complete.json')
            assert d['complete'] and d['provenance']=={**f.provenance(root),'checkpoint':train_done['checkpoint_sha256'],'arm':arm,'seed':seed}
            rr=[];cc=[];nd=0
            for path in sorted((folder/'fixed').glob('*/summary.json')):
                r,n=f.audit_panel(path.parent);assert r['provenance']==d['provenance'];rr.append(r);arrays+=n
                fixed[arm,seed,r['task']]=r['conditions'][m['primary_head']+'__memory']['auc']
            for path in sorted((folder/'diagnostics').glob('*/*/*/summary.json')):
                r,n=f.audit_panel(path.parent);assert r['provenance']==d['provenance'];arrays+=n;nd+=1
            for path in sorted((folder/'closed').glob('*/*/trace.json')):
                r=audit_closed(path.parent);assert r['provenance']==d['provenance'];cc.append(r)
                if r['variant']=='full':closed[arm,seed,r['result']['task']]=r['result']
            assert len(rr)==d['fixed_scenes'] and len(cc)==d['closed_cases'] and nd==d['diagnostic_cases']
            out[f'{arm}_s{seed}']={'fixed':d['metrics'],'closed':{v:{k:float(np.mean([r['result'][k] for r in cc if r['variant']==v]))
                for k in ['success9','success20','base_success9','base_success20','log_area20','base_log_area20']}
                for v in sorted({r['variant'] for r in cc})}}
    for left,right in [(ARMS[1],ARMS[3]),(ARMS[2],ARMS[4]),(ARMS[1],ARMS[0]),(ARMS[1],ARMS[2])]:
        diffs=[]
        for (a,s,t),v in fixed.items():
            if a==left and v is not None and fixed[right,s,t] is not None:diffs.append(v-fixed[right,s,t])
        cc=[{k:r[k]-closed[right,s,t][k] for k in ['success9','success20','log_area20']}
            for (a,s,t),r in closed.items() if a==left]
        paired[left+'__minus__'+right]={'fixed_auc_pairs':len(diffs),'mean_auc_delta':float(np.mean(diffs)) if diffs else None,
                                      'closed_mean_delta':{k:float(np.mean([r[k] for r in cc])) for k in ['success9','success20','log_area20']}}
    f.write_json(root/('WITHIN_SUMMARY.json' if within else 'SUMMARY.json'),{'complete':True,'audit_passed':True,
        'provenance':f.provenance(root),'arrays_verified':arrays,'per_model':out,'paired':paired,'limits':m['limits']})
    print('FIVE_ARM_SUMMARY_COMPLETE',within,arrays,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--root',type=Path,required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--within',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    if a.stage=='prepare':prepare(a.root)
    elif a.stage=='cache':cache(a.root,a.index)
    elif a.stage=='smoke':train(a.root,ARMS[a.index],0,True)
    elif a.stage=='smoke_summary':smoke_summary(a.root)
    elif a.stage=='train':train(a.root,ARMS[a.index%5],a.index//5)
    elif a.stage=='prepare_within':f.prepare_within(a.root)
    elif a.stage=='eval':evaluate(a.root,ARMS[a.index%5],a.index//5,a.within)
    elif a.stage=='summary':summarize(a.root,a.within)
    else:raise ValueError(a.stage)


if __name__=='__main__':main()
