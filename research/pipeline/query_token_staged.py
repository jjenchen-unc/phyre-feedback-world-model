"""Q0 predictor (full base-sized budget), then a frozen-reader residual updater."""
import argparse
import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import query_token_dynamic as q
from query_token_predictor import QueryTokenPredictor

f = q.f
UPDATER = ('residual_in', 'action_in', 'time', 'space', 'state_norm',
           'observation_norm', 'update_attention', 'ff_norm', 'update_ff')
PARENT = f.PROJECT / 'results/query_token_dynamic_20261005'
PACK = '/mnt/meg/jjenchen/phyre-unc/packs/cross_f1_train_s16r16_pca128'
DEV = '/mnt/meg/jjenchen/phyre-unc/packs/cross_f1_dev_pca128'


def is_updater(name):
    return name.split('.')[0] in UPDATER


class Model(q.Model):
    def __init__(self, backbone, din, architecture, stage='q0'):
        QueryTokenPredictor.__init__(self, backbone, din, **architecture)
        self.arm = 'staged_' + stage
        self.stage = stage
        self.use_memory = stage != 'q0'
        for name, p in self.named_parameters():
            p.requires_grad_((not is_updater(name)) if stage == 'q0' else is_updater(name))

    def state_memory(self, state, observed=True):
        # No empty-history bypass: Q0 is a real input even before any observation.
        return state, torch.ones(len(state), self.n_tokens, device=state.device, dtype=torch.bool)

    def forward(self, z0, action, memory=None, mask=None):
        if memory is None:
            memory, mask = self.state_memory(self.initial_state(len(z0)))
        return QueryTokenPredictor.forward(self, z0, action, memory, mask)

    def reset_updater_identity(self):
        for layer in [self.update_attention.out_proj, self.update_ff[-1]]:
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def update_state(self, state, residual, action):
        if self.stage == 'zero':
            residual = torch.zeros_like(residual)
        return QueryTokenPredictor.update_state(self, state, residual, action)


class Q0View(torch.nn.Module):
    """Evaluation reference for a staged model; never invokes its updater."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z0, action):
        return self.model(z0, action)


def fingerprint(model, updater=False):
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        if is_updater(name) == updater:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def epoch_indices(n, seed, epoch):
    g = torch.Generator().manual_seed(20261005 + seed * 100 + epoch)
    return torch.randperm(n, generator=g)[:n // 8 * 8].reshape(-1, 8).sort(1).values.numpy()


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'manifest.json').exists()
    m = f.read(PARENT / 'manifest.json')
    assert f.read(PARENT/'DATA_AUDIT.json')['provenance']=={'manifest':f.digest(PARENT/'manifest.json'),'code':m['code']}
    assert all(f.digest(PARENT/'code'/name)==digest for name,digest in m['code'].items())
    for path, digest in m['source_hashes'].items():assert f.digest(path)==digest
    store=f.Store(include_features=False)
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest()==m['store_files_sha256']
    for row in m['rows']:
        assert not set(row['failures']) & set(row['query'])
        assert not store.labels[row['failures']].any()
        assert all(str(store.tasks[i])==row['task'] for i in row['failures']+row['query'])
    source = f.Pack(PACK)
    assert len(source.actions) == 52276 and len(source.by) == 1600
    assert len({t[:5] for t in source.by}) == 16
    cross_tasks = {r['task'] for r in m['final_panels'] if r['panel'] == 'cross_natural'}
    assert not cross_tasks & set(source.by)
    assert not {t[:5] for t in cross_tasks} & {t[:5] for t in source.by}
    assert len(source.lat[0]) == 52276 and source.lat[0].shape[1:] == (32, 196, 128)
    m.update(run_root=str(root), parent_manifest_sha256=f.digest(PARENT / 'manifest.json'),
             scope='Staged full-budget Q0 predictor then updater-only; existing dev panels, not new test',
             code=f.code_hashes(), stage1_pack=PACK, stage1_dev_pack=DEV,
             stage1_meta_sha256=f.digest(PACK + '_meta.npz'),
             stage1_dev_meta_sha256=f.digest(DEV + '_meta.npz'),
             stage1={'epochs':8, 'batch':8, 'n_train':52276, 'updates':52272,
                     'presentations':418176, 'initialization':'original pretrained predictor; fresh active Q0/read adapters'},
             stage2={'episodes':20000, 'batch':2, 'updates':10000, 'branches':['residual','zero'],
                     'freeze':'backbone,Q0,read adapters,encoder,probes', 'updater_prefixes':list(UPDATER)},
             evaluation={'closed_budget':9, 'scenes_per_split':100, 'stress_budget':20,
                         'stress_cross_per_template':5, 'stress_within_per_template':2,
                         'heads':['original',m['primary_head']], 'diagnostic_prefixes':[1,2,4,8]})
    m['limits'] = ['Stage1 adds a full base-sized budget on top of original pretraining; geometry objective differs from historical base L1',
        'Stage1 train contains1600 scenes including stage2 calibration; calibration is updater-heldout only',
        'Stage21200 scenes; random failure replay, not on-policy', 'Existing cross/within panels reused development evidence',
        'S@9 on all100scenes primary; S@20 only fixed20scenes, length stress',
        'Zero-residual removes information and affects effective parameter gradients; report paired seeds',
        'Three seeds and few templates do not establish universal generalization']
    draws = {}
    for seed in m['seeds']:
        indices = [epoch_indices(52276, seed, ep) for ep in range(8)]
        assert all(v.shape == (6534,8) and len(np.unique(v)) == 52272 for v in indices)
        draws[str(seed)] = hashlib.sha256(b''.join(v.tobytes() for v in indices)).hexdigest()
    f.write_json(root / 'manifest.json', m)
    f.write_json(root / 'DATA_AUDIT.json', {'passed':True,'provenance':f.provenance(root),
        'stage1_scenes':1600,'stage2_scenes':1200,'stage1_epoch_draw_sha256':draws,
        'stage1_cross_template_disjoint':True,'parent_data_audit':f.digest(PARENT / 'DATA_AUDIT.json')})
    f.prepare_within(root)


def checked(root):
    m = f.read(root / 'manifest.json')
    assert m['code'] == f.code_hashes()
    assert f.read(root / 'DATA_AUDIT.json')['provenance'] == f.provenance(root)
    assert f.digest(m['base']) == m['base_sha256'] and f.digest(m['probe']) == m['probe_sha256']
    return m


def model_for(root, stage, seed, checkpoint=None, parent_checkpoint=None):
    m = checked(root)
    ref, probe, rule, pc, _ = f.load_models(memory=None)
    ref.eval().requires_grad_(False)
    probe.eval().requires_grad_(False)
    torch.manual_seed(seed + 7919)
    model = Model(copy.deepcopy(ref), pc['Din'], m['architecture'], stage).cuda()
    if stage != 'q0':
        if parent_checkpoint is None:
            parent = f.read(root / 'models' / f'q0_s{seed}' / 'complete.json')
            assert parent['provenance'] == f.provenance(root) and parent['steps'] == 52272
            parent_checkpoint = Path(parent['checkpoint'])
            assert f.digest(parent_checkpoint) == parent['checkpoint_sha256']
        ck = torch.load(parent_checkpoint, map_location='cpu', weights_only=False)
        assert ck['stage'] == 'q0' and ck['seed'] == seed and ck['provenance'] == f.provenance(root)
        own = model.state_dict()
        # Updater is fresh and identical across stage2 branches, not inherited from stage1.
        own.update({k:v for k,v in ck['state_dict'].items() if not is_updater(k)})
        model.load_state_dict(own)
        model.reset_updater_identity()
    if checkpoint:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert ck['stage'] == stage and ck['seed'] == seed and ck['provenance'] == f.provenance(root)
        model.load_state_dict(ck['state_dict'])
    return ref, probe, rule, model


def save(root, folder, model, seed, step):
    path = folder / 'checkpoints' / f'step_{step:06d}.pt'
    f.atomic_torch(path, {'stage':model.stage, 'seed':seed, 'steps':step,
        'provenance':f.provenance(root), 'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}})
    return path


def target_times(ids):
    return [sorted(np.random.default_rng([20261005,int(i),73]).choice(np.arange(1,32),8,replace=False).tolist()) for i in ids]


def q0_loss(model, probe, source, ids):
    z = torch.from_numpy(source.z(ids)).cuda().float()
    a = torch.from_numpy(source.actions[ids]).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(z[:,0].contiguous(), a).float()
    l = (pred[:,1:] - z[:,1:]).abs().mean()
    g = f.geometry_loss(probe, pred, [source.feature(i) for i in ids], target_times(ids))
    return l + .1*g, l, g


@torch.no_grad()
def dev_l1(model, source, limit=None):
    model.eval()
    total = 0.
    n = min(len(source.actions),limit) if limit else len(source.actions)
    for start in range(0,n,8):
        ids = list(range(start,min(start+8,n)))
        z = torch.from_numpy(source.z(ids)).cuda().float()
        a = torch.from_numpy(source.actions[ids]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred = model(z[:,0].contiguous(),a).float()
        total += float((pred[:,1:]-z[:,1:]).abs().mean()) * len(ids)
    return {'n':n,'latent_l1':total/n,'role':'historical base dev; not unseen test'}


def audit_trainability(model):
    trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all((not is_updater(n)) if model.stage == 'q0' else is_updater(n) for n in trainable)
    return trainable


def train(root, stage, seed, smoke=False, parent_checkpoint=None):
    m = checked(root)
    if not smoke:
        assert f.read(root/'SMOKE.json')['passed'] and f.read(root/'SMOKE.json')['provenance'] == f.provenance(root)
    folder = root / ('smoke' if smoke else 'models') / f'{stage}_s{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        done=f.read(folder/'complete.json')
        assert done['provenance']==f.provenance(root) and f.digest(done['checkpoint'])==done['checkpoint_sha256']
        return done
    if stage!='q0' and parent_checkpoint is None:
        parent=f.read(root/'models'/f'q0_s{seed}'/'complete.json')
        assert parent['provenance']==f.provenance(root) and parent['steps']==52272
        parent_checkpoint=Path(parent['checkpoint'])
        assert f.digest(parent_checkpoint)==parent['checkpoint_sha256']
    ref,probe,rule,model=model_for(root,stage,seed,parent_checkpoint=parent_checkpoint)
    names=audit_trainability(model)
    source=f.Pack(m['stage1_pack']) if stage=='q0' else f.Store()
    if stage=='q0':
        assert f.digest(m['stage1_pack']+'_meta.npz') == m['stage1_meta_sha256']
        dev=f.Pack(m['stage1_dev_pack'])
        assert f.digest(m['stage1_dev_pack']+'_meta.npz') == m['stage1_dev_meta_sha256']
    rows=[r for r in m['rows'] if r['split']=='train']
    total=100 if smoke else m['stage1' if stage=='q0' else 'stage2']['updates']
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.05)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-4,total_steps=total,pct_start=.03)
    frozen_before=fingerprint(model,updater=(stage=='q0'))
    reference_before=f.frozen_fingerprint(ref);probe_before=f.frozen_fingerprint(probe)
    zcheck=torch.from_numpy(source.z(list(range(2)))).cuda().float()
    acheck=torch.from_numpy(source.actions[:2]).cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        initial_prediction=model(zcheck[:,0].contiguous(),acheck).detach().clone()
        if stage=='q0':assert torch.equal(initial_prediction,ref(zcheck[:,0].contiguous(),acheck))
        else:
            state=model.initial_state(2)
            updated=model.update_state(state,zcheck-initial_prediction.float(),acheck)
            assert torch.equal(updated,state)
    initial_model=f.frozen_fingerprint(model)
    start=0;logs=[];draws=[];resume=folder/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['provenance']==f.provenance(root) and ck['total']==total and ck['stage']==stage and ck['seed']==seed
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['steps'];logs=ck['logs'];draws=ck['draws']
        torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif stage=='q0':
        f.write_json(folder/'dev_initial.json',dev_l1(model,dev,16 if smoke else None))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.time();running=[]
    epoch_order=None;grad_checks={}
    for step in range(start,total):
        model.train();ref.eval();probe.eval()
        if stage=='q0':
            epoch,offset=divmod(step,6534)
            if epoch_order is None or offset==0:epoch_order=epoch_indices(52276,seed,epoch)
            ids=epoch_order[offset].tolist()
            loss,l,g=q0_loss(model,probe,source,ids);draws.append(ids)
        else:
            eps=[q.episode(rows,seed,2*step+j) for j in range(2)]
            loss,l,g=q.batch_loss(model,probe,eps,source);draws.extend(eps)
        opt.zero_grad(set_to_none=True)
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        assert torch.isfinite(norm) and norm>0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        if step<3:
            grad_checks[str(step)]={name:float(p.grad.abs().sum()) for name,p in model.named_parameters() if p.grad is not None}
        opt.step();sched.step();running.append([float(l.detach()),float(g.detach()),float(loss.detach())])
        done=step+1
        if done%100==0 or done==total:
            torch.cuda.synchronize()
            row={'steps':done,'losses':np.mean(running,axis=0).tolist(),'seconds_this_attempt':time.time()-t0,
                 'start_step':start,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,'lr':sched.get_last_lr()[0]}
            logs.append(row);running=[];f.write_json(folder/'progress.json',row)
            print('TRAIN',stage,seed,json.dumps(row),flush=True)
        epoch_end=stage=='q0' and done%6534==0
        milestone=done==total or epoch_end or (stage!='q0' and done in [500,2000,5000])
        if milestone:
            save(root,folder,model,seed,done)
            if stage=='q0':f.write_json(folder/f'dev_{done:06d}.json',dev_l1(model,dev,16 if smoke else None))
            elif not smoke:
                calibration(root,folder,model,ref,probe,rule,source,done,
                    {**f.provenance(root),'stage':stage,'seed':seed,'steps':done})
        if done%500==0 or milestone:
            assert fingerprint(model,updater=(stage=='q0'))==frozen_before
            f.atomic_torch(resume,{'provenance':f.provenance(root),'stage':stage,'seed':seed,'total':total,'steps':done,
                'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                'logs':logs,'draws':draws,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
    assert f.frozen_fingerprint(ref)==reference_before and f.frozen_fingerprint(probe)==probe_before
    assert fingerprint(model,updater=(stage=='q0'))==frozen_before
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        if stage!='q0':assert torch.equal(initial_prediction,model(zcheck[:,0].contiguous(),acheck))
    if smoke:
        assert grad_checks['1']['initial_tokens']>0 if stage=='q0' else grad_checks['1']['update_attention.out_proj.weight']>0
        if stage=='residual':assert grad_checks['2']['residual_in.weight']>0
    path=save(root,folder,model,seed,total)
    result={'complete':True,'stage':stage,'seed':seed,'steps':total,'provenance':f.provenance(root),
        'checkpoint':str(path),'checkpoint_sha256':f.digest(path),'logs':logs,'gradient_checks':grad_checks,
        'trainable_names':names,'frozen_sha256':frozen_before,'initial_model_sha256':initial_model,
        'draw_sha256':hashlib.sha256(json.dumps(draws,sort_keys=True).encode()).hexdigest(),
        'parent_checkpoint':str(parent_checkpoint) if parent_checkpoint else None,
        'parent_checkpoint_sha256':f.digest(parent_checkpoint) if parent_checkpoint else None,
        'checks':{'frozen_unchanged':True,'initial_identity':True,'q0_unchanged_stage2':stage!='q0'}}
    f.write_json(folder/'complete.json',result)
    return result


def smoke(root):
    first=train(root,'q0',0,True)
    parent=Path(first['checkpoint'])
    results=[first]
    for stage in ['residual','zero']:
        results.append(train(root,stage,0,True,parent))
    assert results[1]['initial_model_sha256']==results[2]['initial_model_sha256']
    assert results[1]['draw_sha256']==results[2]['draw_sha256']
    assert results[1]['frozen_sha256']==results[2]['frozen_sha256']
    m=checked(root);source=f.Pack(m['cross_pack'])
    for stage in ['residual','zero']:
        ck=Path(next(r['checkpoint'] for r in results if r['stage']==stage))
        ref,probe,rule,model=model_for(root,stage,0,ck,parent)
        heads,hh=f.load_heads(m,f.PROBES,probe);head=heads[m['primary_head']]
        prov={**f.provenance(root),'stage':stage,'smoke':True}
        mixed=next(s for s in m['final_panels'] if s['panel']=='cross_natural' and s['task']=='00000:050')
        panel(root/'smoke'/f'{stage}_s0'/'panel',mixed,source,Q0View(model),model,{m['primary_head']:head},rule,prov)
        difficult=next(s for s in m['final_panels'] if s['panel']=='cross_natural' and s['task']=='00007:047')
        path=root/'smoke'/f'{stage}_s0'/'closed'
        d=q.closed_case(path,difficult,source,model,Q0View(model),head,rule,prov,budget=9)
        assert d['result']['updates']>=1, 'GPU smoke must exercise a real observed failure'
        with np.load(path/'snapshots.npz') as raw:np.testing.assert_array_equal(raw['scores'][0],raw['base_scores'])
    f.write_json(root/'SMOKE.json',{'passed':True,'provenance':f.provenance(root),
        'scope':'GPU100updates per stage; frozen/Q0, paired init, nonempty latent/scorer metrics, closed causal replay and raw audits',
        'stages':{r['stage']:{'logs':r['logs'],'checks':r['checks']} for r in results}})
    print('STAGED_SMOKE_PASS',flush=True)



@torch.no_grad()
def panel(folder,spec,source,ref,model,heads,rule,prov,variant='full'):
    if (folder/'summary.json').exists():
        r=f.read(folder/'summary.json');assert r['provenance']==prov;return r
    folder.mkdir(parents=True,exist_ok=True)
    ids,ctx=spec['indices'],spec['context'];query=spec['query']
    assert not set(query)&set(ctx) and not source.labels[ctx].any()
    assert all(str(source.tasks[i])==spec['task'] for i in ids+ctx)
    model.eval();state=q.observed_state(model,source,ctx,variant)
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
                own=model(z[:,0].contiguous(),a).float()
                assert torch.equal(ep,own) # Q0 path, including frozen learned read adapters
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
            'provenance':prov,'empty_identity':True,'empty_reference':'active Q0 predictor', 'comparison_reference':'original base' if model.stage=='q0' else 'same checkpoint Q0',
            'future_input_isolation':True}
    f.write_json(folder/'summary.json',result);f.audit_panel(folder)
    return result

@torch.no_grad()
def static_case(folder, spec, source, model, ref, head, rule, prov):
    if (folder/'result.json').exists():
        d=f.read(folder/'result.json');assert d['provenance']==prov;return d
    folder.mkdir(parents=True, exist_ok=True)
    ids=spec['indices'];z0=np.array(source.lat[0][ids,0]);actions=source.actions[ids]
    initial=[source.feature(i)[:1].copy() for i in ids]
    original=q.scores(ref,None,False,z0,actions,initial,head,rule)
    q0=q.scores(Q0View(model),None,False,z0,actions,initial,head,rule)
    labels=source.labels[ids]
    def rank(scores):
        hits=np.flatnonzero(labels[np.argsort(-scores,kind='stable')])
        return int(hits[0])+1 if len(hits) else None
    d={'task':spec['task'],'provenance':prov,'original_first_rank':rank(original),'q0_first_rank':rank(q0)}
    f.save_npz(folder/'scores.npz',{'indices':np.asarray(ids),'labels':labels,'original':original,'q0':q0})
    f.write_json(folder/'result.json',d);return d


def calibration(root, folder, model, ref, probe, rule, source, step, prov):
    m=checked(root)
    heads, hashes=f.load_heads(m,f.PROBES,probe)
    assert all(hashes[h]==v for h,v in m['heads'].items())
    rows=[]
    for spec in [s for s in m['final_panels'] if s['panel']=='calibration']:
        ids=sorted(sorted(spec['query'],key=lambda i:f.stable(('staged_cal',spec['task'],i)))[:64])
        spec=dict(spec,indices=ids,query=ids)
        rows.append(panel(folder/'calibration'/str(step)/spec['task'].replace(':','_'),spec,source,
            Q0View(model),model,{m['primary_head']:heads[m['primary_head']]},rule,prov))
    f.write_json(folder/'calibration'/str(step)/'SUMMARY.json',{'provenance':prov,
        'role':'stage2 held-out templates; seen by stage1','metrics':f.aggregate(rows,m['primary_head'])})


def evaluate(root, stage, seed, within):
    m=checked(root)
    done=f.read(root/'models'/f'{stage}_s{seed}'/'complete.json')
    assert done['provenance']==f.provenance(root)
    assert done['steps']==m['stage1' if stage=='q0' else 'stage2']['updates']
    ck=Path(done['checkpoint']);assert f.digest(ck)==done['checkpoint_sha256']
    prov={**f.provenance(root),'checkpoint':f.digest(ck),'stage':stage,'seed':seed}
    folder=root/('within_evaluation' if within else 'evaluation')/f'{stage}_s{seed}'
    if (folder/'complete.json').exists():
        assert f.read(folder/'complete.json')['provenance']==prov;return
    original,probe,rule,model=model_for(root,stage,seed,ck);model.eval()
    heads,hashes=f.load_heads(m,f.PROBES,probe)
    assert all(hashes[h]==v for h,v in m['heads'].items())
    if within:
        inputs=f.read(root/'within_inputs.json');assert inputs['provenance']==f.provenance(root)
        assert f.digest(inputs['pack']+'_meta.npz')==inputs['meta_sha256']
        source=f.Pack(inputs['pack']);specs=inputs['specs']
    else:
        assert f.digest(m['cross_pack']+'_meta.npz')==m['cross_meta_sha256']
        source=f.Pack(m['cross_pack']);specs=[s for s in m['final_panels'] if s['panel']=='cross_natural']
    assert len(specs)==100
    ref=original if stage=='q0' else Q0View(model)
    head=heads[m['primary_head']]
    closed=[];t0=time.time()
    for spec in specs:
        name=spec['task'].replace(':','_')
        if stage=='q0':
            closed.append(static_case(folder/'closed'/name,spec,source,model,original,head,rule,prov))
        else:
            path=folder/'closed'/name
            d=q.closed_case(path,spec,source,model,ref,head,rule,prov,budget=9)
            with np.load(path/'snapshots.npz') as raw:
                np.testing.assert_array_equal(raw['scores'][0],raw['base_scores'])
                q0path=root/('within_evaluation' if within else 'evaluation')/f'q0_s{seed}'/'closed'/name/'scores.npz'
                with np.load(q0path) as baseline:
                    np.testing.assert_array_equal(raw['indices'],baseline['indices'])
                    np.testing.assert_array_equal(raw['base_scores'],baseline['q0'])
            closed.append(d)
        print('PRIMARY',stage,seed,within,spec['task'],round(time.time()-t0),flush=True)
    f.write_json(folder/'PRIMARY_COMPLETE.json',{'provenance':prov,'n':100,'primary':'S@9',
        'reference':'original base' if stage=='q0' else 'same seed stage1 Q0',
        'results':[d if stage=='q0' else d['result'] for d in closed]})
    rows=[]
    evalheads={h:heads[h] for h in m['evaluation']['heads']}
    for spec in specs:
        rows.append(panel(folder/'fixed'/spec['task'].replace(':','_'),spec,source,ref,model,evalheads,rule,prov))
    chosen=f.choose_tasks(specs,2 if within else 5,'ablation_closed')
    diagnostic_count=0
    if stage!='q0':
        for spec in chosen:
            name=spec['task'].replace(':','_')
            q.closed_case(folder/'stress'/name,spec,source,model,ref,head,rule,prov,budget=20)
            # Shared original-base policy prefix, stopped at success. Labels enter only after selection.
            baseline=root/('within_evaluation' if within else 'evaluation')/f'q0_s{seed}'/'closed'/name/'scores.npz'
            with np.load(baseline) as raw:order=np.argsort(-raw['original'],kind='stable')
            ctx=[]
            for local in order:
                i=spec['indices'][int(local)]
                if source.labels[i] or len(ctx)==8:break
                ctx.append(i)
            if not ctx:continue
            query=sorted(sorted([i for i in spec['indices'] if i not in ctx],key=lambda i:f.stable(('staged_diag',spec['task'],i)))[:64])
            for k in sorted({min(k,len(ctx)) for k in [1,2,4,8]}):
                for variant in ['full','zero','static']:
                    ss=dict(spec,context=ctx[:k],indices=query,query=query)
                    panel(folder/'diagnostics'/name/str(k)/variant,ss,source,ref,model,{m['primary_head']:head},rule,prov,variant)
                    diagnostic_count+=1
    f.write_json(folder/'complete.json',{'complete':True,'provenance':prov,'fixed_scenes':100,'closed_scenes':100,
        'stress_scenes':len(chosen) if stage!='q0' else 0,'diagnostic_cases':diagnostic_count,
        'fixed_metrics':{h:f.aggregate(rows,h) for h in evalheads},'seconds':time.time()-t0})


def summarize(root,within):
    m=checked(root);base=root/('within_evaluation' if within else 'evaluation');out={};paired={}
    for seed in m['seeds']:
        for stage in ['q0','residual','zero']:
            folder=base/f'{stage}_s{seed}';done=f.read(folder/'complete.json')
            assert done['provenance']['manifest']==f.provenance(root)['manifest'] and done['closed_scenes']==100
            results=f.read(folder/'PRIMARY_COMPLETE.json')['results']
            def solved(rank,k):return rank is not None and 0<rank<=k
            if stage=='q0':
                rr={r['task']:{'s1':solved(r['q0_first_rank'],1),'s9':solved(r['q0_first_rank'],9),
                              'original_s9':solved(r['original_first_rank'],9)} for r in results}
            else:
                rr={r['task']:{'s1':solved(r['first_success'],1),'s9':r['success9']} for r in results}
                for p in (folder/'closed').glob('*/trace.json'):q.audit_closed(p.parent)
            assert len(rr)==100
            for p in (folder/'fixed').glob('*/summary.json'):f.audit_panel(p.parent)
            for p in (folder/'diagnostics').glob('*/*/*/summary.json'):f.audit_panel(p.parent)
            for p in (folder/'stress').glob('*/trace.json'):q.audit_closed(p.parent)
            out[f'{stage}_s{seed}']={'counts':{k:sum(r[k] for r in rr.values()) for k in next(iter(rr.values()))},
                'n':100,'per_task':rr,'fixed_metrics':done['fixed_metrics']}
        for left,right in [('residual','q0'),('residual','zero')]:
            a=out[f'{left}_s{seed}']['per_task'];b=out[f'{right}_s{seed}']['per_task']
            paired[f'{left}_minus_{right}_s{seed}']={k:sum(int(a[t][k])-int(b[t][k]) for t in a)/100 for k in ['s1','s9']}
            assert paired[f'{left}_minus_{right}_s{seed}']['s1']==0
    f.write_json(root/('WITHIN_SUMMARY.json' if within else 'SUMMARY.json'),{'complete':True,'provenance':f.provenance(root),
        'raw_audit_passed':True,'per_model':out,'paired':paired,'limits':m['limits']})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command');parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['q0','residual','zero'],default='q0')
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--within',action='store_true')
    a=parser.parse_args();torch.set_num_threads(4)
    if a.command=='prepare':prepare(a.root)
    elif a.command=='smoke':smoke(a.root)
    elif a.command=='train':train(a.root,a.stage,a.seed)
    elif a.command=='eval':evaluate(a.root,a.stage,a.seed,a.within)
    elif a.command=='summary':summarize(a.root,a.within)
    else:raise ValueError(a.command)


if __name__=='__main__':main()
