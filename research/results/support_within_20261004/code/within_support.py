"""Evaluation-only within-template extension of the frozen support pilot."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import surface_dist, rule_score
from redesign_lib import load_models, context_memory, digest, write_json, PROJECT
from probe_adaptation import readout, save_npz
from memory_geometry_followup import load_heads
from support_utility import CrossStore, grouped_partition, classify, stable_int, ARMS, SEED
from attribution_a_metrics import describe, log_score, auc

PARENT = PROJECT/'results/support_utility_20261004'
PROBES = PROJECT/'results/probe_adaptation_cross_20261001'
PCA = Path('/mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32_pca128/pca.pt')


def read(path):return json.loads(Path(path).read_text())
def paths(root,smoke):
    sel=read(root/'selection.json');assets=Path(sel['assets'])
    prefix='smoke_' if smoke else ''
    return sel,assets,prefix,str(assets/(prefix+'packs')/'within_support_pca128')
def source_hashes():return {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
def parent_inputs():
    p=read(PARENT/'manifest.json')
    assert digest(p['base'])==p['base_sha256'] and digest(p['probe'])==p['probe_sha256']
    return p


@torch.no_grad()
def prefilter(root,shard,nshards,smoke):
    from transformers import AutoModel
    sel,assets,prefix,_=paths(root,smoke);parent=parent_inputs()
    enc=AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large',trust_remote_code=True,
        revision='e831a0347737fcaa660b39c57d41c109de399845').eval().cuda().half()
    base,probe,rule,_,_=load_models()
    pc=torch.load(PCA,map_location='cpu',weights_only=False);mu=pc['mean'].cuda();comp=pc['components'].cuda()
    palette=torch.tensor([[255,255,255],[243,79,70],[107,206,187],[24,119,242],
        [75,74,164],[185,202,210],[0,0,0],[252,223,227]],device='cuda').float()/255
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    out=assets/(prefix+'scores');out.mkdir(parents=True,exist_ok=True);t0=time.time()
    candidates=sorted((assets/(prefix+'candidates')).glob('*.npz'))[shard::nshards]
    for path in candidates:
        target=out/path.name
        if target.exists():continue
        with np.load(path) as c:
            codes=c['frames'];actions=c['action'];f0=c['feats0'];aids=c['action_idx'];scores=[];initial=[]
            dyn=[o for o in range(f0.shape[1]) if f0[0,o,13]<.5 and f0[0,o,11]<.5]
            for start in range(0,len(codes),32):
                cc=codes[start:start+32];x=palette[torch.from_numpy(cc.astype(np.int64)).cuda()].flip(1).permute(0,3,1,2)
                x=((F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)-mean)/std).half()
                z=enc(pixel_values=torch.stack([x,x],2)).last_hidden_state[:,1:197]
                z=((z.half().float()-mu)@comp).half();initial.append(z.cpu().numpy())
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=base(z.float(),torch.from_numpy(actions[start:start+32]).cuda()).float()
                b,nt,ns,dim=pred.shape;kv=probe.inp(pred.reshape(b*nt,ns,dim))+probe.space
                ff=np.repeat(f0[start:start+b,None],nt,axis=1).copy()
                for o in dyn:
                    q=torch.from_numpy(np.repeat(f0[start:start+b,o],nt,axis=0)).cuda()
                    xy,ang=probe.read(kv,q);xy=xy.reshape(b,nt,2).cpu().numpy();ang=ang.reshape(b,nt,2).cpu().numpy()
                    ff[:,:,o,:2]=xy/256;ff[:,:,o,2]=(np.arctan2(ang[...,1],ang[...,0])/2)/(2*np.pi)%1
                scores.extend(rule_score(surface_dist(f),rule) for f in ff)
            scores=np.asarray(scores);k=8 if smoke else 1024;assert len(scores)>=k and np.isfinite(scores).all()
            order=np.argsort(-scores,kind='stable')[:k];initial=np.concatenate(initial)
            save_npz(target,{'action_idx':aids[order],'action':actions[order],'initial_latent':initial[order],
                'full_action_idx':aids,'full_scores':scores,'shortlist_positions':order})
            write_json(target.with_suffix('.json'),{'candidate_sha256':digest(path),'selection_sha256':digest(root/'selection.json'),
                'base':parent['base_sha256'],'probe':parent['probe_sha256'],'pca':digest(PCA),'scores_sha256':digest(target),
                'code':source_hashes(),'rule':'historical original-probe vectorized legacy-tanh prefilter; no outcome input'})
        print('SHORTLIST',path.stem,len(scores),round(time.time()-t0),flush=True)
    write_json(root/f'{prefix}prefilter_complete_{shard}.json',{'complete':True,'files':[p.name for p in candidates],'nshards':nshards})


def cache_provenance(root,pack,parent):
    return {'selection':digest(root/'selection.json'),'meta':digest(pack+'_meta.npz'),
        'base':parent['base_sha256'],'heads':parent['heads'],'original_head':parent['probe_sha256'],'code':source_hashes()}


@torch.no_grad()
def base_cache(root,shard,nshards,smoke):
    sel,assets,prefix,pack=paths(root,smoke);parent=parent_inputs();store=CrossStore(pack)
    base,original,rule,_,_=load_models();heads,hh=load_heads(parent,PROBES,original)
    assert all(hh[h]==v for h,v in parent['heads'].items())
    prov=cache_provenance(root,pack,parent);t0=time.time()
    tasks=sorted(store.by)[shard::nshards]
    for task in tasks:
        folder=root/(prefix+'base_cache')/task.replace(':','_');folder.mkdir(parents=True,exist_ok=True)
        if (folder/'summary.json').exists():assert read(folder/'summary.json')['provenance']==prov;continue
        ids=store.by[task];assert len(ids)==(8 if smoke else 1024)
        fs=[store.feature(i) for i in ids];y=store.labels[ids]
        truth={'indices':np.asarray(ids),'labels':y,'query_mask':np.ones(len(ids),bool),
            'distance':np.stack([surface_dist(f) for f in fs]),'xy':np.stack([f[:,:,:2]*256 for f in fs])}
        values={h:defaultdict(list) for h in heads};l1=[];l2=[]
        for start in range(0,len(ids),32):
            ii=ids[start:start+32];z=torch.from_numpy(store.z(ii)).cuda().float();act=torch.from_numpy(store.actions[ii]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=base(z[:,0].contiguous(),act).float()
            diff=pred[:,1:]-z[:,1:];l1.extend(diff.abs().mean((1,2,3)).cpu().tolist());l2.extend(diff.square().mean((1,2,3)).sqrt().cpu().tolist())
            for h,head in heads.items():
                v=readout(head,pred,fs[start:start+32],rule)
                for key,a in v.items():values[h][key].append(a)
        for h,fields in values.items():
            v={k:np.concatenate(a) for k,a in fields.items()};v['latent_l1']=np.asarray(l1,np.float32);v['latent_l2_rms']=np.asarray(l2,np.float32)
            save_npz(folder/(h+'.npz'),v)
        save_npz(folder/'truth.npz',truth)
        write_json(folder/'summary.json',{'task':task,'provenance':prov,'rule':rule,'indices':ids})
        print('BASE_CACHE',task,round(time.time()-t0),flush=True)
    write_json(root/f'{prefix}base_complete_{shard}.json',{'complete':True,'provenance':prov,'tasks':tasks,'nshards':nshards})


def prepare(root,smoke):
    sel,assets,prefix,pack=paths(root,smoke);parent=parent_inputs();store=CrossStore(pack)
    path=root/(prefix+'manifest.json');assert not path.exists()
    assert read(root/'EXPOSURE_AUDIT.json')['passed']
    assert read(root/'EXPOSURE_AUDIT.json')['selection_sha256']==digest(root/'selection.json')
    expected=([next(t for t in sel['tasks'] if t.startswith(tp+':')) for tp in [sel['templates'][0],sel['templates'][-1]]] if smoke else sel['tasks'])
    assert sorted(store.by)==sorted(expected)
    rows=[];cache_hashes={};prov=cache_provenance(root,pack,parent)
    for task in sorted(store.by):
        ids=store.by[task];roles,groups=grouped_partition(store,task,ids,{})
        folder=root/(prefix+'base_cache')/task.replace(':','_');check=read(folder/'summary.json');assert check['provenance']==prov
        with np.load(folder/'predicted_mix_s0.npz') as b:scores=b['log_score']
        assert len(ids)==len(scores)==(8 if smoke else 1024)
        # Verify simulation/pack actions are exactly the shortlist, with no outcome pruning.
        with np.load(assets/(prefix+'scores')/(task.replace(':','_')+'.npz')) as b:
            selected={int(i):a for i,a in zip(b['action_idx'],b['action'])}
        packed_ids=[int(Path(str(store.files[i])).stem.split('_')[-1]) for i in ids]
        assert set(packed_ids)==set(selected)
        for i,ai in zip(ids,packed_ids):np.testing.assert_array_equal(store.actions[i],selected[ai])
        failed=[i for i in roles['support'] if not store.labels[i]]
        minima={i:float(surface_dist(store.feature(i))[1:].min()) for i in failed}
        row={'task':task,'indices':ids,**roles,'groups':groups,'failures':failed,
            'N':[i for i in failed if minima[i]<=5],'far':[i for i in failed if minima[i]>=20]}
        row=classify(row,[scores[ids.index(i)] for i in roles['support']],store.labels)
        first=ids[int(np.argmax(scores))];row['natural_context']=[] if store.labels[first] else [first]
        row['first_action_success']=bool(store.labels[first]);rows.append(row)
        cache_hashes[task]={p.name:digest(p) for p in folder.glob('*') if p.is_file()}
    models={}
    for seed in [0,1,2]:
        for arm in ARMS:
            name=f'{arm}_s{seed}';p=PARENT/'models'/name/'memory.pt';done=read(p.parent/'complete.json')
            assert done['complete'] and done['steps']==2000 and done['provenance']['manifest']==sel['parent_manifest_sha256']
            assert digest(p)==done['checkpoint_sha256']
            models[name]={'path':str(p),'sha256':digest(p),'training_complete_sha256':digest(p.parent/'complete.json')}
    m={'scope':sel['scope'],'selection_sha256':digest(root/'selection.json'),'parent_manifest_sha256':sel['parent_manifest_sha256'],
        'models':models,'rows':rows,'pack':pack,'meta_sha256':digest(pack+'_meta.npz'),
        'base':parent['base'],'base_sha256':parent['base_sha256'],'probe':parent['probe'],'probe_sha256':parent['probe_sha256'],
        'heads':parent['heads'],'primary_head':'predicted_mix_s0','cache_hashes':cache_hashes,'code':source_hashes(),
        'coverage':{'scenes':len(rows),'natural_failed':sum(bool(r['natural_context']) for r in rows),'selected_eligible':sum(r['eligible'] for r in rows),
            'selected_by_template':dict(Counter(r['task'][:5] for r in rows if r['eligible']))},
        'smoke':smoke,'seeds':[0,1,2],'arms':ARMS}
    write_json(path,m);print('PREPARED',json.dumps(m['coverage']),flush=True)


def specs(m):
    out=[]
    for row in m['rows']:
        out.append(('within_natural',row,'natural',row['natural_context']))
        if row['eligible']:
            for cat in ['N','H','F']:
                donor=min(row[cat],key=lambda i:stable_int(f'{SEED}:donor:{row["task"]}:{i}'))
                out.append(('within_selected',row,cat,[donor]))
    return out


@torch.no_grad()
def evaluate(root,arm,seed,smoke):
    _,_,prefix,_=paths(root,smoke);m=read(root/(prefix+'manifest.json'));name=f'{arm}_s{seed}'
    assert m['code']==source_hashes();mc=m['models'][name];assert digest(mc['path'])==mc['sha256']
    if not smoke:assert read(root/'smoke_check.json')['passed']
    assert digest(m['pack']+'_meta.npz')==m['meta_sha256']
    base,original,rule,_,model=load_models(memory=mc['path']);heads,hh=load_heads(m,PROBES,original)
    assert all(hh[h]==v for h,v in m['heads'].items());source=CrossStore(m['pack'])
    prov={'manifest':digest(root/(prefix+'manifest.json')),'checkpoint':mc['sha256'],'code':source_hashes(),'heads':hh}
    out=root/(prefix+'evaluation')/name;t0=time.time()
    for panel,row,condition,context in specs(m):
        task=row['task'];folder=out/panel/task.replace(':','_')/condition;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'summary.json').exists():assert read(folder/'summary.json')['provenance']==prov;continue
        ids=row['indices'];query=[i for i in ids if i not in context] if condition=='natural' else row['query']
        assert not set(query)&set(context) and not source.labels[context].any()
        reference=root/(prefix+'base_cache')/task.replace(':','_')
        for f,h in m['cache_hashes'][task].items():assert digest(reference/f)==h
        with np.load(reference/'truth.npz') as t:truth={k:t[k].copy() for k in t.files}
        truth['query_mask']=np.isin(ids,query);mask=truth['query_mask'];labels=truth['labels']
        np.testing.assert_array_equal(truth['indices'],ids);np.testing.assert_array_equal(labels,source.labels[ids])
        bpdata={}
        for h in heads:
            with np.load(reference/(h+'.npz')) as p:bpdata[h]={k:p[k].copy() for k in p.files}
        fs=[source.feature(i) for i in ids];mem,mm=context_memory(model,source,context)
        collected={h:defaultdict(list) for h in heads};l1=[];l2=[];empty_checked=False
        for start in range(0,len(ids),32):
            ii=ids[start:start+32];features=fs[start:start+32];z=torch.from_numpy(source.z(ii)).cuda().float();act=torch.from_numpy(source.actions[ii]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(z[:,0].contiguous(),act,mem.expand(len(ii),-1,-1),mm.expand(len(ii),-1)).float()
            assert torch.equal(pred[:,0],z[:,0]);diff=pred[:,1:]-z[:,1:]
            l1.extend(diff.abs().mean((1,2,3)).cpu().tolist());l2.extend(diff.square().mean((1,2,3)).sqrt().cpu().tolist())
            for h,head in heads.items():
                value=readout(head,pred,features,rule)
                for k,v in value.items():collected[h][k].append(v)
                if start==0:
                    altered=[f.copy() for f in features]
                    for f in altered:f[1:]=123
                    changed=readout(head,pred,altered,rule)
                    for k in value:np.testing.assert_array_equal(value[k],changed[k])
            if start==0:
                em,emask=context_memory(model,source,[])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    bp=base(z[:,0].contiguous(),act).float()
                    ep=model(z[:,0].contiguous(),act,em.expand(len(ii),-1,-1),emask.expand(len(ii),-1)).float()
                assert torch.equal(ep,bp);empty_checked=True
                for h,head in heads.items():
                    check=readout(head,bp,features,rule)
                    for k,v in check.items():np.testing.assert_array_equal(v,bpdata[h][k][:len(ii)])
        green=int(np.flatnonzero(fs[0][0,:,9]>.5)[0]);goals=[o for o in range(fs[0].shape[1]) if fs[0][0,o,10]>.5 and fs[0][0,o,13]<.5 and fs[0][0,o,11]<.5]
        metrics={}
        for h in heads:
            value={k:np.concatenate(v) for k,v in collected[h].items()};value['latent_l1']=np.asarray(l1,np.float32);value['latent_l2_rms']=np.asarray(l2,np.float32)
            if not context:
                for k in value:np.testing.assert_array_equal(value[k],bpdata[h][k])
            for kind,v in [('base',bpdata[h]),('memory',value)]:
                key=h+'__'+kind;save_npz(folder/(key+'.npz'),v)
                mmx=describe(v,truth,labels,mask,rule,green,goals,base=bpdata[h] if kind=='memory' else None)
                delta=v['log_score']-bpdata[h]['log_score']
                for label,use in [('success',mask&labels),('failure',mask&~labels)]:mmx[label+'_score_change']=float(delta[use].mean()) if use.any() else None
                metrics[key]=mmx
        save_npz(folder/'truth.npz',truth)
        write_json(folder/'summary.json',{'task':task,'panel':panel,'condition':condition,'context':context,'query':query,
            'rule':rule,'green':green,'dynamic_goals':goals,'conditions':metrics,'provenance':prov,
            'empty_identity':empty_checked,'future_input_isolation':True,'base_first_batch_replay':True})
        print(name,panel,task,condition,round(time.time()-t0),flush=True)
    write_json(out/'complete.json',{'complete':True,'provenance':prov,'specs':len(specs(m))})


def summarize(root,smoke):
    _,_,prefix,_=paths(root,smoke);m=read(root/(prefix+'manifest.json'));assert m['code']==source_hashes()
    names=['uniform_s0'] if smoke else [f'{a}_s{s}' for s in [0,1,2] for a in ARMS]
    results={};arrays=0
    for name in names:
        folder=root/(prefix+'evaluation')/name;done=read(folder/'complete.json');assert done['complete'] and done['specs']==len(specs(m))
        assert done['provenance']['manifest']==digest(root/(prefix+'manifest.json')) and done['provenance']['checkpoint']==m['models'][name]['sha256']
        rows=[]
        for panel,spec,condition,ctx in specs(m):
            path=folder/panel/spec['task'].replace(':','_')/condition;r=read(path/'summary.json')
            assert r['provenance']==done['provenance'] and r['context']==ctx and r['empty_identity'] and r['future_input_isolation']
            with np.load(path/'truth.npz') as t:truth={k:t[k] for k in t.files}
            mask=truth['query_mask'];y=truth['labels'];assert truth['indices'][mask].tolist()==r['query']
            for key,metrics in r['conditions'].items():
                with np.load(path/(key+'.npz')) as v:
                    np.testing.assert_allclose(log_score(v['distance'],r['rule']),v['log_score'],rtol=1e-12,atol=1e-12)
                    assert auc(y[mask],v['log_score'][mask])==metrics['auc']
                    mm=describe(v,truth,y,mask,r['rule'],r['green'],r['dynamic_goals'])
                    for field in ['auc','first_success_rank','hit_at_10','all','success','failure']:assert mm[field]==metrics[field]
                    h,kind=key.split('__');assert metrics['all']['latent_l1']==r['conditions']['original__'+kind]['all']['latent_l1']
                    with np.load(path/(h+'__base.npz')) as b:
                        delta=v['log_score']-b['log_score']
                        for label,use in [('success',mask&y),('failure',mask&~y)]:
                            assert metrics[label+'_score_change']==(float(delta[use].mean()) if use.any() else None)
                    arrays+=1
            rows.append(r)
        results[name]=rows
    if smoke:
        write_json(root/'smoke_check.json',{'passed':True,'manifest_sha256':digest(root/'smoke_manifest.json'),'arrays_verified':arrays,'code':source_hashes()})
        print('WITHIN_SMOKE_PASSED',arrays,flush=True);return
    key=m['primary_head']+'__memory';bkey=m['primary_head']+'__base';macros={};paired={}
    for panel,condition in sorted({(p,c) for p,_,c,_ in specs(m)}):
        tag=panel+'/'+condition;macros[tag]={};paired[tag]={}
        for name,rows in results.items():
            rr=[r for r in rows if r['panel']==panel and r['condition']==condition and (panel!='within_natural' or r['context'])]
            valid=[r for r in rr if r['conditions'][key]['auc'] is not None]
            macros[tag][name]={'scenes':len(rr),'auc_scenes':len(valid),
                'auc':float(np.mean([r['conditions'][key]['auc'] for r in valid])) if valid else None,
                'base_auc':float(np.mean([r['conditions'][bkey]['auc'] for r in valid])) if valid else None,
                'per_task':[{'task':r['task'],'auc':r['conditions'][key]['auc'],'base_auc':r['conditions'][bkey]['auc']} for r in valid]}
        for a,b in [('near','uniform'),('mixed','uniform'),('mixed','near')]:
            by=defaultdict(list);seedmeans={}
            for seed in [0,1,2]:
                aa=macros[tag][f'{a}_s{seed}']['per_task'];bb=macros[tag][f'{b}_s{seed}']['per_task'];dd=[]
                assert [r['task'] for r in aa]==[r['task'] for r in bb]
                for x,y in zip(aa,bb):
                    assert x['base_auc']==y['base_auc'];d=x['auc']-y['auc'];by[x['task']].append(d);dd.append(d)
                seedmeans[str(seed)]=float(np.mean(dd)) if dd else None
            tasks=sorted(by);rng=np.random.default_rng(SEED);samples=np.zeros(10000)
            for tp in sorted({t[:5] for t in tasks}):
                vals=np.array([np.mean(by[t]) for t in tasks if t[:5]==tp]);samples+=rng.choice(vals,(10000,len(vals)),replace=True).sum(1)
            paired[tag][a+'-'+b]={'n_scenes':len(tasks),'mean':float(np.mean([np.mean(by[t]) for t in tasks])) if tasks else None,
                'ci95_conditional_on_templates_and_models':np.quantile(samples/len(tasks),[.025,.975]).tolist() if tasks else None,
                'per_seed':seedmeans,'by_template':{tp:float(np.mean([np.mean(by[t]) for t in tasks if t[:5]==tp])) for tp in sorted({t[:5] for t in tasks})}}
    result={'complete':True,'audit_passed':True,'arrays_verified':arrays,'manifest_sha256':digest(root/'manifest.json'),
        'scope':m['scope'],'coverage':m['coverage'],'macros':macros,'paired':paired}
    write_json(root/'SUMMARY.json',result);print('WITHIN_SUMMARY_COMPLETE',arrays,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prefilter','base','prepare','eval','summary']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=1);p.add_argument('--smoke',action='store_true')
    p.add_argument('--arm',choices=ARMS,default='uniform');p.add_argument('--seed',type=int,default=0);a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prefilter':prefilter(a.root,a.shard,a.nshards,a.smoke)
    elif a.stage=='base':base_cache(a.root,a.shard,a.nshards,a.smoke)
    elif a.stage=='prepare':prepare(a.root,a.smoke)
    elif a.stage=='eval':evaluate(a.root,a.arm,a.seed,a.smoke)
    else:summarize(a.root,a.smoke)


if __name__=='__main__':main()
