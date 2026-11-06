"""Fresh memory with L1+geometry; no R0 weights and no Near-based selection."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from common import MemoryPredictor, surface_dist
from redesign_lib import Store, load_models, context_memory, digest, write_json, PROJECT
from memory_geometry_followup import geometry_loss, frozen_fingerprint, load_heads
from probe_adaptation import readout, save_npz
from attribution_a_metrics import describe, auc, log_score

PARENT = PROJECT/'results/support_utility_20261004'
WITHIN = PROJECT/'results/support_within_20261004'
PROBES = PROJECT/'results/probe_adaptation_cross_20261001'
SEED = 20261005


def read(p): return json.loads(Path(p).read_text())
def stable(x): return int(hashlib.sha256(str(x).encode()).hexdigest()[:16], 16)
def code_hashes(): return {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
def provenance(root): return {'manifest': digest(root/'manifest.json'), 'code': code_hashes()}


class Pack:
    def __init__(self, prefix):
        self.lat = [np.load(prefix+'_latents.npy', mmap_mode='r')]
        with np.load(prefix+'_meta.npz') as d: self.meta = {k:d[k] for k in d.files}
        self.tasks, self.actions = self.meta['tasks'], self.meta['action']
        self.labels = self.meta['solved'].astype(bool)
        self.by = {str(t): np.flatnonzero(self.tasks==t).tolist() for t in sorted(set(self.tasks))}
    def z(self, ids): return np.array(self.lat[0][ids])
    def feature(self, i): return self.meta['features'][i][:,self.meta['mask'][i]]


def choose_tasks(rows, n, tag):
    return [r for tp in sorted({r['task'][:5] for r in rows})
            for r in sorted([r for r in rows if r['task'][:5]==tp], key=lambda r:stable((SEED,tag,r['task'])))[:n]]


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/'manifest.json').exists(), 'Never replace a registered manifest'
    parent, inv = read(PARENT/'manifest.json'), read(PARENT/'inventory.json')
    assert digest(PARENT/'inventory.json')==parent['inventory_sha256']
    store=Store(include_features=False)
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest()==inv['store_files_sha256']
    for p,h in inv['source_hashes'].items(): assert digest(p)==h
    rows=[]
    for r in inv['rows']:
        failures=[i for i in r['support'] if not store.labels[i]]
        assert failures==r['failures']
        if not failures or not r['query']: continue
        assert not set(r['support'])&set(r['query'])
        assert not {r['groups'][str(i)]['group'] for i in r['support']} & {r['groups'][str(i)]['group'] for i in r['query']}
        assert all(str(store.tasks[i])==r['task'] for i in failures+r['query'])
        assert not {tuple(store.actions[i]) for i in failures} & {tuple(store.actions[i]) for i in r['query']}
        rows.append({'task':r['task'], 'split':r['split'], 'failures':failures, 'query':r['query'],
                     'times':[sorted(np.random.default_rng([SEED,i,73]).choice(np.arange(1,32),8,replace=False).tolist()) for i in r['query']]})
    train=[r for r in rows if r['split']=='train']; cal=[r for r in rows if r['split']=='calibration']
    assert len(train)==1200 and len({r['task'][:5] for r in train})==12
    counts={s:{'scenes':len(rr),'by_template':dict(Counter(r['task'][:5] for r in rr)),
               'queries':sum(len(r['query']) for r in rr), 'positive_queries':sum(int(store.labels[r['query']].sum()) for r in rr),
               'failed_supports':sum(len(r['failures']) for r in rr)} for s,rr in [('train',train),('calibration',cal)]}
    panels=[]
    for panel,rr in [('train_seen',choose_tasks(train,2,'train')),('calibration',choose_tasks(cal,8,'calibration'))]:
        for r in rr:
            context=[min(r['failures'],key=lambda i:stable((SEED,'donor',r['task'],i)))]
            panels.append({'panel':panel,'task':r['task'],'context':context,'query':r['query'],'indices':r['query']})
    cross=[]
    for r in parent['cross_rows']:
        ids=r['indices'];ctx=r['natural_context']
        cross.append({'panel':'cross_natural','task':r['task'],'context':ctx,'indices':ids,'query':[i for i in ids if i not in ctx]})
    curve=choose_tasks([r for r in panels if r['panel']=='train_seen'],2,'curve_train')+choose_tasks([r for r in panels if r['panel']=='calibration'],2,'curve_cal')+choose_tasks(cross,2,'curve_cross')
    curve=[dict(r,indices=sorted(sorted(r['query'],key=lambda i:stable((SEED,'curve_query',r['task'],i)))[:64])) for r in curve]
    for r in curve:r['query']=r['indices']
    m={k:parent[k] for k in ['base','base_sha256','probe','probe_sha256','heads','cross_pack','cross_meta_sha256']}
    m.update({'scope':'Fresh memory only; all valid scenes, uniform K1 failures and uniform queries; no N/H/F filtering; fixed final checkpoint.',
              'initialization':'New MemoryPredictor constructor; zero output projections; NO old memory state loaded',
              'inventory_sha256':digest(PARENT/'inventory.json'), 'parent_manifest_sha256':digest(PARENT/'manifest.json'),
              'source_hashes':inv['source_hashes'],'store_files_sha256':inv['store_files_sha256'],
              'architecture':{'mode':'residual','mem_steps':16,'mem_grid':10,'per_attempt':True,'gate_empty':True},
              'training':{'steps':20000,'queries':8,'lr':3e-4,'weight_decay':.05,'clip':1.,'geometry_weight':.1,'scheduler':'OneCycleLR,3%warmup'},
              'seeds':[0,1,2],'checkpoints':[0,500,2000,5000,10000,20000], 'primary_head':'predicted_mix_s0',
              'rows':rows,'curve_panels':curve,'final_panels':panels+cross,'counts':counts,'code':code_hashes(),
              'within_selection_sha256':digest(WITHIN/'selection.json'),
              'limits':['Reused cross D0; no test-set claim','Train_seen queries are training data, not validation',
                        'Calibration templates seen by base but held out from this memory',
                        'Curves use fixed <=64-query subsets, not directly comparable to full final AUC',
                        'No best-on-cross checkpoint selection','No closed-loop success claim','No causal comparison to older different-data R0 runs']})
    write_json(root/'manifest.json',m);write_json(root/'DATA_AUDIT.json',{'passed':True,'manifest_sha256':digest(root/'manifest.json'),'counts':counts,
        'checks':['all1200training scenes retained','failure labels','task identity','role and group disjointness','no identical support/query actions','source hashes'],'near_filter_used':False})
    print('PREPARED',json.dumps(counts),flush=True)


def models(m, seed, checkpoint=None):
    assert digest(m['base'])==m['base_sha256'] and digest(m['probe'])==m['probe_sha256']
    base,probe,rule,pc,_=load_models(memory=None)
    base.eval().requires_grad_(False);probe.eval().requires_grad_(False)
    torch.manual_seed(seed)
    model=MemoryPredictor(base,pc['Din'],**m['architecture']).cuda()
    if checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['base']==m['base'] and ck['initialization']=='fresh'
        missing,extra=model.load_state_dict(ck['state_dict'],strict=False)
        assert not extra and all(k.startswith('base.') for k in missing)
    heads,hh=load_heads(m,PROBES,probe)
    assert all(hh[h]==v for h,v in m['heads'].items())
    return base,probe,rule,model,heads


def episode(rows, seed, step):
    rng=np.random.default_rng([SEED,seed,step,0]);r=rows[int(rng.integers(len(rows)))]
    slots=rng.choice(len(r['query']),min(8,len(r['query'])),replace=False).tolist()
    srng=np.random.default_rng([SEED,seed,step,1]);ctx=[int(srng.choice(r['failures']))]
    return r,[r['query'][j] for j in slots],[r['times'][j] for j in slots],ctx


def weights(model):return {k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('base.')}
def atomic_torch(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)


def checkpoint(folder,model,m,seed,step,prov):
    ck={**m['architecture'],'base':m['base'],'base_sha256':m['base_sha256'],'unfreeze':False,
        'state_dict':weights(model),'initialization':'fresh','seed':seed,'step':step,'provenance':prov}
    path=folder/'checkpoints'/f'step_{step:06d}.pt';atomic_torch(path,ck);return path


@torch.no_grad()
def evaluate_panel(folder,spec,source,base,model,heads,rule,prov,identity=False,replay=None):
    if (folder/'summary.json').exists():
        result=read(folder/'summary.json');assert result['provenance']==prov;return result
    folder.mkdir(parents=True,exist_ok=True);model.eval();base.eval()
    ids,ctx=spec['indices'],spec['context'];query=spec['query']
    assert not set(query)&set(ctx) and all(not source.labels[i] for i in ctx)
    assert all(str(source.tasks[i])==spec['task'] for i in ids+ctx)
    fs=[source.feature(i) for i in ids];labels=source.labels[ids];qset=set(query);mask=np.array([i in qset for i in ids])
    truth={'indices':np.asarray(ids),'labels':labels,'query_mask':mask,'distance':np.stack([surface_dist(f) for f in fs]),'xy':np.stack([f[:,:,:2]*256 for f in fs])}
    mem,mm=context_memory(model,source,ctx);col=defaultdict(lambda:defaultdict(list));isolation=False;empty=False
    for start in range(0,len(ids),32):
        ii=ids[start:start+32];z=torch.from_numpy(source.z(ii)).cuda().float();act=torch.from_numpy(source.actions[ii]).cuda()
        z0=z[:,0].contiguous()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            bp=base(z0,act).float();mp=model(z0,act,mem.expand(len(ii),-1,-1),mm.expand(len(ii),-1)).float()
        if identity or not ctx:assert torch.equal(bp,mp)
        assert torch.equal(mp[:,0],z[:,0])
        if start==0:
            em,emask=context_memory(model,source,[])
            with torch.autocast('cuda',dtype=torch.bfloat16):ep=model(z0,act,em.expand(len(ii),-1,-1),emask.expand(len(ii),-1)).float()
            assert torch.equal(ep,bp);empty=True
        for kind,pred in [('base',bp),('memory',mp)]:
            diff=pred[:,1:]-z[:,1:];l1=diff.abs().mean((1,2,3)).cpu().numpy();l2=diff.square().mean((1,2,3)).sqrt().cpu().numpy()
            for h,head in heads.items():
                v=readout(head,pred,fs[start:start+32],rule)
                if start==0 and kind=='memory':
                    altered=[f.copy() for f in fs[:len(ii)]]
                    for f in altered:f[1:]=123
                    changed=readout(head,pred,altered,rule)
                    for k in v:np.testing.assert_array_equal(v[k],changed[k])
                    isolation=True
                v.update(latent_l1=l1,latent_l2_rms=l2)
                for k,a in v.items():col[h+'__'+kind][k].append(a)
    raw={k:{field:np.concatenate(v) for field,v in dd.items()} for k,dd in col.items()}
    green=int(np.flatnonzero(fs[0][0,:,9]>.5)[0]);goals=[o for o in range(fs[0].shape[1]) if fs[0][0,o,10]>.5 and fs[0][0,o,13]<.5 and fs[0][0,o,11]<.5]
    metrics={}
    for k,v in raw.items():
        h,kind=k.split('__')
        if replay and kind=='base':
            with np.load(Path(replay)/(h+'__base.npz')) as old:
                for field in ['log_score','xy','distance']:np.testing.assert_array_equal(v[field],old[field])
        save_npz(folder/(k+'.npz'),v)
        metrics[k]=describe(v,truth,labels,mask,rule,green,goals,base=raw[h+'__base'] if kind=='memory' else None)
    save_npz(folder/'truth.npz',truth)
    result={**spec,'rule':rule,'green':green,'dynamic_goals':goals,'conditions':metrics,'provenance':prov,
            'initial_nonempty_identity':identity,'empty_identity':empty,'future_input_isolation':isolation,'historical_base_replay':bool(replay)}
    write_json(folder/'summary.json',result);return result


def audit_panel(folder):
    r=read(folder/'summary.json')
    with np.load(folder/'truth.npz') as t:truth={k:t[k] for k in t.files}
    y,mask=truth['labels'],truth['query_mask'];assert truth['indices'][mask].tolist()==r['query']
    for k,metrics in r['conditions'].items():
        with np.load(folder/(k+'.npz')) as v:
            np.testing.assert_allclose(log_score(v['distance'],r['rule']),v['log_score'],rtol=1e-12,atol=1e-12)
            a=auc(y[mask],v['log_score'][mask]);assert a==metrics['auc']
            mm=describe(v,truth,y,mask,r['rule'],r['green'],r['dynamic_goals'])
            for field in ['auc','first_success_rank','hit_at_10','all','success','failure']:assert mm[field]==metrics[field]
    assert r['empty_identity'] and r['future_input_isolation'];return r,len(r['conditions'])


def curves(root,folder,m,seed,step,ck,base,model,heads,rule,store,cross,smoke):
    prov={**provenance(root),'checkpoint':digest(ck),'step':step,'seed':seed}
    specs=m['curve_panels']
    if smoke:
        specs=[next(r for r in specs if r['panel']==p) for p in ['train_seen','calibration','cross_natural']]
        specs=[dict(r,indices=r['indices'][:4],query=r['indices'][:4]) for r in specs]
    h={k:heads[k] for k in ['original',m['primary_head']]};records=[];count=0
    for spec in specs:
        path=folder/'curves'/f'{step:06d}'/spec['panel']/spec['task'].replace(':','_')
        evaluate_panel(path,spec,cross if spec['panel']=='cross_natural' else store,base,model,h,rule,prov,identity=step==0)
        rec,n=audit_panel(path);records.append(rec);count+=n
    out=aggregate(records,m['primary_head'])
    write_json(folder/'curves'/f'{step:06d}'/'SUMMARY.json',{'step':step,'seed':seed,'audit_passed':True,'arrays_verified':count,'groups':out,'provenance':prov})
    print('CURVE',seed,step,json.dumps(out),flush=True)


def aggregate(rows,head):
    out={}
    for panel in sorted({r['panel'] for r in rows}):
        allrows=[r for r in rows if r['panel']==panel];rr=[r for r in allrows if not panel.endswith('_natural') or r['context']]
        valid=[r for r in rr if r['conditions'][head+'__memory']['auc'] is not None]
        def mean(group,kind,field):
            vs=[]
            for r in group:
                v=r['conditions'][head+'__'+kind]
                for k in field.split('/'):v=v[k]
                if v is not None:vs.append(v)
            return float(np.mean(vs)) if vs else None
        out[panel]={'total_scenes':len(allrows),'nonempty_scenes':len(rr),'auc_scenes':len(valid),
             'base_auc':mean(valid,'base','auc'),'memory_auc':mean(valid,'memory','auc'),
             'base_l1':mean(rr,'base','all/latent_l1'),'memory_l1':mean(rr,'memory','all/latent_l1'),
             'base_success_surface_mae':mean(rr,'base','success/surface_mae'),'memory_success_surface_mae':mean(rr,'memory','success/surface_mae')}
    return out


def train(root,seed,smoke):
    m=read(root/'manifest.json');assert m['code']==code_hashes();prov=provenance(root)
    assert read(root/'DATA_AUDIT.json')['manifest_sha256']==prov['manifest']
    if not smoke:assert read(root/'SMOKE.json')['passed'] and read(root/'SMOKE.json')['provenance']==prov
    folder=root/('smoke' if smoke else 'models')/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():assert read(folder/'complete.json')['provenance']==prov;return
    store=Store();cross=Pack(m['cross_pack']);assert digest(m['cross_pack']+'_meta.npz')==m['cross_meta_sha256']
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest()==m['store_files_sha256']
    for p,h in m['source_hashes'].items():assert digest(p)==h
    base,probe,rule,model,heads=models(m,seed);initial=frozen_fingerprint(model)
    fixed={k:frozen_fingerprint(v) for k,v in [('base',base),('probe',probe)]}
    params=[p for p in model.parameters() if p.requires_grad];assert params and all(not p.requires_grad for p in base.parameters())
    cfg=m['training'];steps=8 if smoke else cfg['steps'];checks=[0,4,8] if smoke else m['checkpoints']
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,cfg['lr'],total_steps=steps,pct_start=.5 if smoke else .03)
    rows=[r for r in m['rows'] if r['split']=='train'];start=0;draws=[];logs=[];gradient_checks={};resume=folder/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['provenance']==prov and ck['steps']==steps and ck['seed']==seed
        missing,extra=model.load_state_dict(ck['weights'],strict=False);assert not extra and all(k.startswith('base.') for k in missing)
        opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start,draws,logs,gradient_checks=ck['step'],ck['draws'],ck['logs'],ck['gradient_checks']
    if start==0:
        assert all(torch.count_nonzero(o.weight)==0 and torch.count_nonzero(o.bias)==0 for o in model.o)
        ck=checkpoint(folder,model,m,seed,0,prov)
        curves(root,folder,m,seed,0,ck,base,model,heads,rule,store,cross,smoke)
    t0=time.time();running=np.zeros(3);nrun=0
    for step in range(start,steps):
        row,ids,times,ctx=episode(rows,seed,step)
        assert not set(ids)&set(ctx) and len(ctx)==1 and not store.labels[ctx].any()
        z=torch.from_numpy(store.z(ids)).cuda().float();act=torch.from_numpy(store.actions[ids]).cuda();features=[store.feature(i) for i in ids]
        model.train();base.eval();probe.eval();mem,mask=context_memory(model,store,ctx)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z[:,0],act,mem.expand(len(ids),-1,-1),mask.expand(len(ids),-1)).float()
        latent=(pred[:,1:]-z[:,1:]).abs().mean();geo=geometry_loss(probe,pred,features,times);loss=latent+cfg['geometry_weight']*geo
        assert torch.isfinite(loss)
        if step==0:
            g=torch.autograd.grad(geo,pred,retain_graph=True)[0];assert torch.isfinite(g).all() and g.abs().sum()>0 and torch.count_nonzero(g[:,0])==0
            gradient_checks['geometry_to_prediction']=float(g.norm())
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['clip']);assert torch.isfinite(norm) and norm>0
        if step==0:
            v=sum(float(o.weight.grad.abs().sum()) for o in model.o);assert v>0;gradient_checks['zero_output_projection_gradient']=v
        if step==1:
            v=float(model.zin[1].weight.grad.abs().sum());assert v>0;gradient_checks['encoder_gradient_after_first_update']=v
        assert all(p.grad is None for p in base.parameters()) and all(p.grad is None for p in probe.parameters())
        opt.step();scheduler.step()
        draws.append({'task':row['task'],'query':ids,'times':times,'context':ctx,'query_successes':int(store.labels[ids].sum())})
        running+=np.array([float(latent.detach()),float(geo.detach()),float(loss.detach())]);nrun+=1
        if (step+1)%100==0 or step+1==steps:
            log={'step':step+1,'mean_latent_l1':running[0]/nrun,'mean_geometry':running[1]/nrun,'mean_loss':running[2]/nrun,'steps_in_average':nrun,'lr':scheduler.get_last_lr()[0],'seconds_this_attempt':time.time()-t0}
            logs.append(log);write_json(folder/'progress.json',log);print('TRAIN',seed,json.dumps(log),flush=True);running*=0;nrun=0
        if step+1 in checks:
            ck=checkpoint(folder,model,m,seed,step+1,prov)
            curves(root,folder,m,seed,step+1,ck,base,model,heads,rule,store,cross,smoke)
        if (step+1)%250==0 or step+1 in checks or step+1==steps:
            atomic_torch(resume,{'weights':weights(model),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'step':step+1,'steps':steps,'seed':seed,'provenance':prov,'draws':draws,'logs':logs,'gradient_checks':gradient_checks})
    assert fixed=={k:frozen_fingerprint(v) for k,v in [('base',base),('probe',probe)]}
    assert initial!=frozen_fingerprint(model)
    final=folder/'checkpoints'/f'step_{steps:06d}.pt'
    write_json(folder/'draws.json',draws)
    write_json(folder/'complete.json',{'complete':True,'steps':steps,'seed':seed,'provenance':prov,'checkpoint':str(final),'checkpoint_sha256':digest(final),
        'initial_model_sha256':initial,'frozen_modules_unchanged':True,'gradient_checks':gradient_checks,'logs':logs,
        'unique_supports':len({i for r in draws for i in r['context']}),'query_successes':sum(r['query_successes'] for r in draws),
        'query_count':sum(len(r['query']) for r in draws),'scenes_seen':len({r['task'] for r in draws})})
    if smoke:
        for r in draws: assert not set(r['query'])&set(r['context'])
        write_json(root/'SMOKE.json',{'passed':True,'provenance':prov,'steps':steps,'gradient_checks':gradient_checks,
            'checks':['fresh zero-projection nonempty identity','empty identity after training','geometry and encoder gradients','frozen base/probe','future input isolation','raw AUC/L1/geometry recomputation'],
            'scope':'Engineering checks only, no performance gate'})
        print('FRESH_SMOKE_PASSED',flush=True)


def prepare_within(root):
    m=read(root/'manifest.json');w=read(WITHIN/'manifest.json');assert digest(WITHIN/'selection.json')==m['within_selection_sha256']
    assert w['base_sha256']==m['base_sha256'] and w['heads']==m['heads']
    assert read(WITHIN/'EXPOSURE_AUDIT.json')['passed'] and read(WITHIN/'SETUP_AUDIT.json')['passed']
    assert digest(w['pack']+'_meta.npz')==w['meta_sha256'];store=Pack(w['pack']);specs=[]
    for r in w['rows']:
        ids=r['indices']
        with np.load(WITHIN/'base_cache'/r['task'].replace(':','_')/'predicted_mix_s0.npz') as d:first=ids[int(np.argmax(d['log_score']))]
        ctx=[] if store.labels[first] else [first];assert ctx==r['natural_context']
        specs.append({'panel':'within_natural','task':r['task'],'indices':ids,'context':ctx,'query':[i for i in ids if i not in ctx]})
    out={'specs':specs,'pack':w['pack'],'meta_sha256':w['meta_sha256'],'selection_sha256':m['within_selection_sha256'],
         'source_manifest_sha256':digest(WITHIN/'manifest.json'),'provenance':provenance(root),'N_filter_used':False}
    path=root/'within_inputs.json'
    if path.exists():assert read(path)==out
    else:write_json(path,out)
    print('WITHIN_REGISTERED',len(specs),flush=True)


def evaluate(root,seed,within):
    m=read(root/'manifest.json');prov=provenance(root);assert m['code']==code_hashes()
    done=read(root/'models'/f'seed_{seed}'/'complete.json');assert done['complete'] and done['steps']==m['training']['steps'] and done['provenance']==prov
    ck=Path(done['checkpoint']);assert digest(ck)==done['checkpoint_sha256']
    base,probe,rule,model,heads=models(m,seed,ck);prov={**prov,'checkpoint':digest(ck),'seed':seed}
    if within:
        wm=read(root/'within_inputs.json');assert wm['provenance']==provenance(root) and digest(wm['pack']+'_meta.npz')==wm['meta_sha256']
        specs=wm['specs'];store=Pack(wm['pack']);cross=None;prov['within_inputs']=digest(root/'within_inputs.json')
    else:
        specs=m['final_panels'];store=Store();cross=Pack(m['cross_pack'])
        assert digest(m['cross_pack']+'_meta.npz')==m['cross_meta_sha256']
    folder=root/('within_evaluation' if within else 'evaluation')/f'seed_{seed}';t0=time.time()
    for spec in specs:
        src=cross if spec['panel']=='cross_natural' else store
        replay=PROBES/'evaluation/cross'/spec['task'].replace(':','_') if spec['panel']=='cross_natural' else None
        evaluate_panel(folder/spec['panel']/spec['task'].replace(':','_'),spec,src,base,model,heads,rule,prov,replay=replay)
        print('EVAL',seed,spec['panel'],spec['task'],round(time.time()-t0),flush=True)
    write_json(folder/'complete.json',{'complete':True,'provenance':prov,'specs':len(specs)})


def summarize(root,within):
    m=read(root/'manifest.json');folder=root/('within_evaluation' if within else 'evaluation');allrows=[];arrays=0;per_seed={}
    for seed in m['seeds']:
        done=read(folder/f'seed_{seed}'/'complete.json');assert done['complete']
        train_done=read(root/'models'/f'seed_{seed}'/'complete.json')
        assert done['provenance']['manifest']==digest(root/'manifest.json') and done['provenance']['code']==code_hashes()
        assert done['provenance']['checkpoint']==train_done['checkpoint_sha256']==digest(train_done['checkpoint'])
        expected=read(root/'within_inputs.json')['specs'] if within else m['final_panels'];assert done['specs']==len(expected)
        rows=[]
        for spec in expected:
            r,n=audit_panel(folder/f'seed_{seed}'/spec['panel']/spec['task'].replace(':','_'))
            assert r['provenance']==done['provenance']
            for k in ['task','context','query','indices','panel']:assert r[k]==spec[k]
            arrays+=n;r['seed']=seed;rows.append(r)
        per_seed[str(seed)]={h:aggregate(rows,h) for h in ['original']+list(m['heads'])};allrows+=rows
    paired={}
    for panel in sorted({r['panel'] for r in allrows}):
        by=defaultdict(list)
        for r in allrows:
            if r['panel']!=panel or (panel.endswith('_natural') and not r['context']):continue
            a,b=(r['conditions'][m['primary_head']+'__'+k]['auc'] for k in ['memory','base'])
            if a is not None:by[r['task']].append(a-b)
        tasks=sorted(by);delta=np.array([np.mean(by[t]) for t in tasks]);rng=np.random.default_rng(SEED);samples=np.zeros(10000)
        for tp in sorted({t[:5] for t in tasks}):
            vals=delta[[i for i,t in enumerate(tasks) if t[:5]==tp]];samples+=rng.choice(vals,(10000,len(vals)),replace=True).sum(1)
        paired[panel]={'n_scenes':len(tasks),'memory_minus_base':float(delta.mean()) if tasks else None,
            'conditional_scene_ci95':np.quantile(samples/len(tasks),[.025,.975]).tolist() if tasks else None,
            'by_template':{tp:float(np.mean([np.mean(by[t]) for t in tasks if t[:5]==tp])) for tp in sorted({t[:5] for t in tasks})}}
    write_json(root/('WITHIN_SUMMARY.json' if within else 'SUMMARY.json'),{'complete':True,'audit_passed':True,'arrays_verified':arrays,
        'provenance':provenance(root),'per_seed':per_seed,'paired':paired,'limits':m['limits']})
    print('FRESH_SUMMARY_COMPLETE',arrays,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','train','eval','summary','prepare_within']);p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--within',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    if a.stage=='prepare':prepare(a.root)
    elif a.stage in ['smoke','train']:train(a.root,a.seed,a.stage=='smoke')
    elif a.stage=='eval':evaluate(a.root,a.seed,a.within)
    elif a.stage=='summary':summarize(a.root,a.within)
    elif a.stage=='prepare_within':prepare_within(a.root)
if __name__=='__main__':main()
