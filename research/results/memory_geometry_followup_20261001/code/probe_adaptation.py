"""Frozen-predictor probe adaptation diagnostic, with matched training controls."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import AttnProbe, surface_dist, rule_score
from redesign_lib import Store, load_models, context_memory, digest, write_json, PROJECT, ROOT
from attribution_a_metrics import log_score, describe

ARMS = ['true_control', 'predicted_mix', 'initial_only']
SOURCES = ['base', 'memory', 'observed', 'initial']

def provenance(root):
    return {'manifest': digest(root/'manifest.json'), 'code': {f.name: digest(f) for f in Path(__file__).parent.glob('*.py')}}

def prepare(root):
    oldpath = PROJECT/'results/memory_redesign_20260928/data/dataset.json'
    old = json.loads(oldpath.read_text())
    stagepath = PROJECT/'results/attribution_stage_a_cross_20260930/manifest.json'
    stage = json.loads(stagepath.read_text())
    store = Store(include_features=False)
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest() == old['files_sha256']
    rows = []
    for split, tasks in [('train', old['train_tasks']), ('calibration', old['validation_tasks'])]:
        eligible = [t for t in tasks if old['tasks'][t]['base_trace']['failures']]
        for template in sorted(set(t[:5] for t in eligible)):
            ts = sorted(t for t in eligible if t[:5] == template)
            rng = np.random.default_rng([20261001, int(template), split == 'calibration'])
            if split == 'train': ts = sorted(rng.choice(ts, min(32, len(ts)), replace=False).tolist())
            for task in ts:
                entry = old['tasks'][task]
                rng = np.random.default_rng([20261001, *map(int, task.split(':'))])
                query = sorted(rng.choice(entry['query_pool'], min(16, len(entry['query_pool'])), replace=False).tolist())
                context = entry['base_trace']['failures'][:1]
                assert len(context) == 1 and not store.labels[context[0]] and not set(context)&set(query)
                assert all(store.tasks[i] == task for i in context+query)
                times = [sorted(rng.choice(np.arange(1, 32), 8, replace=False).tolist()) for _ in query]
                rows.append({'task': task, 'split': split, 'query': query, 'context': context, 'times': times})
    train = [r for r in rows if r['split']=='train']; cal = [r for r in rows if r['split']=='calibration']
    assert not set(r['task'] for r in train)&set(r['task'] for r in cal)
    assert not set(r['task'][:5] for r in rows)&set(stage['evaluation_templates'])
    m = {'scope': 'Probe adaptation pilot, fixed R0/base; reused cross D0, not a new test or closed-loop run.',
         'old_data':str(oldpath),'old_data_sha256':digest(oldpath),'store_files_sha256':old['files_sha256'],
         'training_pack_meta_sha256':{str(ROOT/'packs'/(tag+'_meta.npz')):digest(ROOT/'packs'/(tag+'_meta.npz')) for tag in old['packs']},
         'stage_a':str(stagepath.parent),'stage_a_manifest_sha256':digest(stagepath),
         'base':stage['base'],'base_sha256':stage['base_sha256'],'probe':stage['probe'],'probe_sha256':stage['probe_sha256'],
         'memory':stage['models']['initial_R0'],'rows':rows,'seeds':[0,1,2], 'arms':ARMS,
         'training':{'steps':2000,'batch':128,'lr':3e-5,'weight_decay':.05,'clip':1.,'initialization':'original frozen probe',
                     'objective':'original XY L1/256*10 + doubled-angle L1 on bars; no success/ranking loss',
                     'checkpoint_selection':'fixed final step, no cross-based selection'},
         'context_scope':'K1 from old base failure trace on support-half pack; not a stable-1024 deploy pool',
         'holdout_scope':'Calibration is adaptation-heldout and R0-memory-template-heldout; base/original probe have seen these templates.',
         'counts':{'train_scenes':len(train),'calibration_scenes':len(cal),'train_queries':sum(len(r['query']) for r in train),'calibration_queries':sum(len(r['query']) for r in cal),'cross_scenes':len(stage['tasks'])},
         'smoke_train_tasks':[r['task'] for r in train[:2]],
         'smoke_cross_tasks':[next(r['task'] for r in stage['tasks'] if r['task'].startswith(t)) for t in ['00000','00014']]}
    root.mkdir(parents=True,exist_ok=True);write_json(root/'manifest.json',m);print(json.dumps(m['counts']),flush=True)

def load_fixed(m):
    for path,h in [(m['base'],m['base_sha256']),(m['probe'],m['probe_sha256']),(m['memory']['path'],m['memory']['sha256'])]:assert digest(path)==h
    base,probe,rule,pc,model=load_models(memory=m['memory']['path'])
    for module in [base,model,probe]:module.eval().requires_grad_(False)
    return base,probe,rule,pc,model

@torch.no_grad()
def cache(root, smoke):
    m=json.loads((root/'manifest.json').read_text());base,probe,rule,pc,model=load_fixed(m);store=Store()
    for path,h in m['training_pack_meta_sha256'].items():assert digest(path)==h
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest()==m['store_files_sha256']
    rows=[r for r in m['rows'] if r['split']=='train' and (not smoke or r['task'] in m['smoke_train_tasks'])]
    folder=root/('smoke_cache' if smoke else 'cache');folder.mkdir(exist_ok=True)
    prov=provenance(root)
    if (folder/'complete.json').exists():assert json.loads((folder/'complete.json').read_text())['provenance']==prov;return
    n=sum(len(r['query']) for r in rows);shape=(n,8,pc['S'],pc['Din'])
    arrays={s:np.lib.format.open_memmap(folder/(s+'.npy'),mode='w+',dtype=np.float16,shape=shape) for s in ['base','memory','observed']}
    initial=np.lib.format.open_memmap(folder/'initial.npy',mode='w+',dtype=np.float16,shape=(n,pc['S'],pc['Din']))
    metadata=[];offset=0;t0=time.time()
    for row in rows:
        ids=row['query'];z=torch.from_numpy(store.z(ids)).cuda().float();a=torch.from_numpy(store.actions[ids]).cuda()
        mem,mask=context_memory(model,store,row['context'])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            bp=base(z[:,0].contiguous(),a).float();mp=model(z[:,0].contiguous(),a,mem.expand(len(ids),-1,-1),mask.expand(len(ids),-1)).float()
        assert torch.equal(bp[:,0],z[:,0]) and torch.equal(mp[:,0],z[:,0])
        for j,i in enumerate(ids):
            f=store.feature(i);times=row['times'][j]
            for s,pred in [('base',bp),('memory',mp),('observed',z)]:
                v=pred[j,times].cpu().numpy().astype(np.float16);assert np.isfinite(v).all();arrays[s][offset]=v
            initial[offset]=z[j,0].cpu().numpy().astype(np.float16)
            dyn=[o for o in range(f.shape[1]) if f[0,o,13]<.5 and f[0,o,11]<.5];assert dyn
            metadata.append({'task':row['task'],'index':int(i),'times':times,'dynamic':dyn,'features':f.tolist()});offset+=1
        if len(metadata)%160==0:print(json.dumps({'cached_queries':offset,'total':n,'seconds':round(time.time()-t0)}),flush=True)
    assert offset==n
    for v in arrays.values():v.flush()
    initial.flush();write_json(folder/'metadata.json',metadata)
    hashes={f.name:digest(f) for f in folder.glob('*.npy')};hashes['metadata.json']=digest(folder/'metadata.json')
    write_json(folder/'complete.json',{'complete':True,'provenance':prov,'n_queries':n,'hashes':hashes});print('CACHE_COMPLETE',n,flush=True)


def train(root,arm,seed,smoke):
    m=json.loads((root/'manifest.json').read_text());cfg=m['training'];prov=provenance(root)
    folder=root/('smoke_cache' if smoke else 'cache');done=json.loads((folder/'complete.json').read_text());assert done['provenance']==prov
    metadata=json.loads((folder/'metadata.json').read_text());assert digest(folder/'metadata.json')==done['hashes']['metadata.json']
    sources=['base','memory'] if arm=='predicted_mix' else ['observed'] if arm=='true_control' else ['initial']
    arrays={s:np.load(folder/(s+'.npy'),mmap_mode='r') for s in sources}
    features=[np.array(r['features'],np.float32) for r in metadata]
    by={t:[i for i,r in enumerate(metadata) if r['task']==t] for t in sorted(set(r['task'] for r in metadata))};tasks=list(by)
    ck=torch.load(m['probe'],map_location='cpu',weights_only=False);assert digest(m['probe'])==m['probe_sha256']
    torch.manual_seed(seed);rng=np.random.default_rng(seed);probe=AttnProbe(ck['D'],ck['S']).cuda();probe.load_state_dict(ck['state_dict']);probe.train()
    opt=torch.optim.AdamW(probe.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);steps=3 if smoke else cfg['steps']
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    out=root/('smoke_heads' if smoke else 'heads')/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():assert json.loads((out/'complete.json').read_text())['provenance']==prov;return
    log=[];t0=time.time();initial={k:v.clone() for k,v in probe.state_dict().items()};draw_hash=hashlib.sha256()
    for step in range(steps):
        tokens=[];queries=[];xy=[];angles=[];bars=[];draw=[]
        for j in range(cfg['batch']):
            task=tasks[rng.integers(len(tasks))];i=by[task][rng.integers(len(by[task]))];entry=metadata[i]
            slot=int(rng.integers(8));t=entry['times'][slot];o=int(rng.choice(entry['dynamic']));f=features[i]
            src=sources[(j+step)%len(sources)];tokens.append(arrays[src][i] if src=='initial' else arrays[src][i,slot])
            queries.append(f[0,o]);xy.append(f[t,o,:2]*256);angle=f[t,o,2]*2*np.pi;angles.append([np.cos(2*angle),np.sin(2*angle)]);bars.append(f[0,o,4:8].argmax()==1);draw.append((i,slot,o))
        draw_hash.update(np.asarray(draw,np.int64).tobytes())
        tok=torch.from_numpy(np.stack(tokens).astype(np.float32)).cuda();q=torch.from_numpy(np.stack(queries)).cuda();target=torch.tensor(np.asarray(xy),dtype=torch.float32,device='cuda');ang=torch.tensor(angles,dtype=torch.float32,device='cuda');bar=torch.tensor(bars,dtype=torch.float32,device='cuda')
        pred,pa=probe.read(probe.inp(tok)+probe.space,q)
        loss=F.l1_loss(pred,target)/256*10+((pa-ang).abs().mean(-1)*bar).mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(probe.parameters(),cfg['clip']);assert torch.isfinite(norm) and norm>0;opt.step();sched.step()
        if (step+1)%100==0 or step+1==steps:
            rec={'step':step+1,'loss':float(loss.detach()),'grad_norm':float(norm),'seconds':round(time.time()-t0)};log.append(rec);print(arm,seed,json.dumps(rec),flush=True)
    assert any(not torch.equal(v,initial[k]) for k,v in probe.state_dict().items())
    state={k:v.detach().cpu() for k,v in probe.state_dict().items()};tmp=out/'probe.tmp';torch.save({'state_dict':state,'D':ck['D'],'S':ck['S'],'rule':ck['rule'],'provenance':prov,'arm':arm,'seed':seed,'steps':steps,'draw_sha256':draw_hash.hexdigest()},tmp);tmp.replace(out/'probe.pt')
    write_json(out/'complete.json',{'complete':True,'provenance':prov,'arm':arm,'seed':seed,'steps':steps,'draw_sha256':draw_hash.hexdigest(),'checkpoint_sha256':digest(out/'probe.pt'),'log':log})

@torch.no_grad()
def readout(probe,pred,features,rule):
    b,nt,ns,_=pred.shape;kv=probe.inp(pred.float().reshape(b*nt,ns,-1))+probe.space;distances=[];positions=[]
    for j,f in enumerate(features):
        pf=np.repeat(f[:1],nt,0).copy()
        for o in range(f.shape[1]):
            if f[0,o,13]>=.5 or f[0,o,11]>=.5:continue
            q=torch.from_numpy(np.repeat(f[:1,o],nt,0)).cuda();xy,ang=probe.read(kv[j*nt:(j+1)*nt],q)
            pf[:,o,:2]=xy.cpu().numpy()/256;ang=ang.cpu().numpy();pf[:,o,2]=(np.arctan2(ang[:,1],ang[:,0])/2)/(2*np.pi)%1
        distances.append(surface_dist(pf));positions.append(pf[:,:,:2]*256)
    ds=np.stack(distances);return {'distance':ds,'xy':np.stack(positions),'log_score':log_score(ds,rule)}

def save_npz(path,data):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(path)

@torch.no_grad()
def evaluate(root,shard,nshards,smoke):
    m=json.loads((root/'manifest.json').read_text());prov=provenance(root);base,original,rule,pc,model=load_fixed(m)
    stage=json.loads((Path(m['stage_a'])/'manifest.json').read_text());assert digest(Path(m['stage_a'])/'manifest.json')==m['stage_a_manifest_sha256']
    heads={'original':original};seeds=[0] if smoke else m['seeds'];head_hashes={};draws={}
    for seed in seeds:
        for arm in ARMS:
            name=f'{arm}_s{seed}';folder=root/('smoke_heads' if smoke else 'heads')/name;c=json.loads((folder/'complete.json').read_text());assert c['provenance']==prov and digest(folder/'probe.pt')==c['checkpoint_sha256']
            ck=torch.load(folder/'probe.pt',map_location='cpu',weights_only=False);h=AttnProbe(ck['D'],ck['S']).cuda();h.load_state_dict(ck['state_dict']);heads[name]=h.eval().requires_grad_(False);head_hashes[name]=c['checkpoint_sha256'];draws.setdefault(seed,[]).append(c['draw_sha256'])
    assert all(len(set(v))==1 for v in draws.values()),'Unmatched target sampling across arms'
    prov={**prov,'heads':head_hashes}
    store=Store();lat=np.load(stage['pack']+'_latents.npy',mmap_mode='r');assert digest(stage['pack']+'_meta.npz')==stage['meta_sha256']
    assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest()==m['store_files_sha256']
    with np.load(stage['pack']+'_meta.npz') as packed:meta={k:packed[k] for k in packed.files}
    entries=[{'panel':'cross','entry':r} for r in stage['tasks'] if not smoke or r['task'] in m['smoke_cross_tasks']]
    if not smoke:entries += [{'panel':'calibration','entry':r} for r in m['rows'] if r['split']=='calibration']
    entries=entries[shard::nshards];out=root/('smoke_eval' if smoke else 'evaluation');t0=time.time()
    for spec in entries:
        panel=spec['panel'];entry=spec['entry'];task=entry['task'];folder=out/panel/task.replace(':','_');folder.mkdir(parents=True,exist_ok=True)
        if (folder/'summary.json').exists():assert json.loads((folder/'summary.json').read_text())['provenance']==prov;continue
        if panel=='cross':
            ids=entry['indices'];labels=meta['solved'][ids].astype(bool);mask=np.isin(ids,entry['query']);features=[meta['features'][i][:,meta['mask'][i]] for i in ids];context=entry['contexts']['K1']
            zfun=lambda ii:np.stack([lat[i] for i in ii]);afun=lambda ii:meta['action'][ii]
            k=max(1,len(context));cz=torch.zeros((1,k)+lat.shape[1:],device='cuda');ca=torch.zeros(1,k,3,device='cuda');cy=torch.zeros(1,k,device='cuda');cm=torch.zeros(1,k,dtype=torch.bool,device='cuda')
            for j,i in enumerate(context):assert not meta['solved'][i];cz[0,j]=torch.from_numpy(np.array(lat[i])).cuda().float();ca[0,j]=torch.from_numpy(meta['action'][i]).cuda();cm[0,j]=True
            with torch.autocast('cuda',dtype=torch.bfloat16):mem,mm=model.encode_memory(cz,ca,cy,cm,model.context_residual(cz,ca).float())
        else:
            ids=entry['query'];labels=store.labels[ids];mask=np.ones(len(ids),bool);features=[store.feature(i) for i in ids];context=entry['context'];zfun=store.z;afun=lambda ii:store.actions[ii];mem,mm=context_memory(model,store,context)
        green=int(np.flatnonzero(features[0][0,:,9]>.5)[0]);goals=[o for o in range(features[0].shape[1]) if features[0][0,o,10]>.5 and features[0][0,o,13]<.5 and features[0][0,o,11]<.5]
        truth={'distance':np.stack([surface_dist(f) for f in features]),'xy':np.stack([f[:,:,:2]*256 for f in features]),'labels':labels,'query_mask':mask,'indices':np.asarray(ids)}
        collected={};errors={s:[] for s in SOURCES};empty_exact=True;input_boundary_checked=False
        for start in range(0,len(ids),32):
            ii=ids[start:start+32];fs=features[start:start+32];z=torch.from_numpy(zfun(ii)).cuda().float();act=torch.from_numpy(afun(ii)).cuda();z0=z[:,0].contiguous()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                bp=base(z0,act).float();mp=model(z0,act,mem.expand(len(ii),-1,-1),mm.expand(len(ii),-1)).float()
            if not context:assert torch.equal(bp,mp)
            preds={'base':bp,'memory':mp,'observed':z,'initial':z[:,:1].expand_as(z)}
            for source,pred in preds.items():
                assert torch.equal(pred[:,0],z[:,0]);diff=pred[:,1:]-z[:,1:];errors[source].append({'latent_l1':diff.abs().mean((1,2,3)).cpu().numpy(),'latent_l2_rms':diff.square().mean((1,2,3)).sqrt().cpu().numpy()})
                for name,head in heads.items():
                    key=f'{name}__{source}';value=readout(head,pred,fs,rule)
                    for field,v in value.items():collected.setdefault(key,{}).setdefault(field,[]).append(v)
            if not input_boundary_checked:
                altered=[f.copy() for f in fs]
                for f in altered:f[1:]=123
                test=readout(original,bp,altered,rule)
                np.testing.assert_array_equal(test['distance'],collected['original__base']['distance'][-1]);input_boundary_checked=True
        save_npz(folder/'truth.npz',truth);metrics={};replay=[]
        for key,fields in collected.items():
            values={k:np.concatenate(v) for k,v in fields.items()};source=key.split('__')[1]
            for field in ['latent_l1','latent_l2_rms']:values[field]=np.concatenate([v[field] for v in errors[source]])
            if panel=='cross' and key in ['original__base','original__memory','original__observed']:
                oldname={'base':'base','memory':'initial_R0_K1' if context else 'base','observed':'observed_latent'}[source]
                with np.load(Path(m['stage_a'])/'tasks'/task.replace(':','_')/(oldname+'.npz')) as old:
                    for field in ['distance','xy','log_score','latent_l1']:np.testing.assert_array_equal(values[field],old[field])
                replay.append(source)
            save_npz(folder/(key+'.npz'),values);metrics[key]=describe(values,truth,labels,mask,rule,green,goals)
        write_json(folder/'summary.json',{'task':task,'panel':panel,'provenance':prov,'context':context,'query':np.asarray(ids)[mask].tolist(),'rule':rule,'green':green,'dynamic_goals':goals,'conditions':metrics,'original_exact_replay':replay,'input_boundary_checked':input_boundary_checked,'empty_memory_identity':empty_exact,'seconds_cumulative':round(time.time()-t0)})
        print(json.dumps({'panel':panel,'task':task,'conditions':len(metrics),'seconds':round(time.time()-t0)}),flush=True)
    write_json(out/f'complete_{shard}.json',{'complete':True,'provenance':prov,'tasks':[(x['panel'],x['entry']['task']) for x in entries]})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','train','eval']);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=ARMS,default=ARMS[0]);p.add_argument('--seed',type=int,default=0);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=1);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare(a.root)
    elif a.stage=='cache':cache(a.root,a.smoke)
    elif a.stage=='train':train(a.root,a.arm,a.seed,a.smoke)
    else:evaluate(a.root,a.shard,a.nshards,a.smoke)

if __name__=='__main__':main()
