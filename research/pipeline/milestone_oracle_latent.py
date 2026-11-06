"""True future latent + unchanged milestone scorer, exact shared candidate pools.

Reuse verified observation/pack caches. Missing rollouts are prefetched by one
CPU simulator per GPU worker, encoded with the historical encoder, then deleted.
Only scores and audit receipts are retained; no 300-GiB duplicate latent cache.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

P=Path('/path/to/phyre-unc');D=Path('/mnt/meg/jjenchen/phyre-unc')
R=P/'results/milestone_oracle_latent_20261015';M=P/'results/milestone_memory_20261015'
OLD=P/'results/small_official_20261010';PCA=D/'latents/levjepa_overlap_g32_pca128/pca.pt'
PHYRE='/path/to/research-assets/miniconda3/envs/phyre/bin/python'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def npz(path,**data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(path)
def provenance():return {'plan':sha(R/'PLAN.json'),'code':{str(f):sha(f) for f in sorted((R/'code').glob('*.py'))}}
def checked():
    plan=read(R/'PLAN.json')
    for path,digest in plan['dependencies'].items():assert sha(path)==digest,path
    for path,spec in plan['pack_files'].items():
        stat=Path(path).stat();assert stat.st_size==spec['size'] and stat.st_mtime_ns==spec['mtime_ns'],path
    return plan


def prepare():
    assert not (R/'PLAN.json').exists()
    parent=read(M/'PLAN.json');dependencies={str(M/'PLAN.json'):sha(M/'PLAN.json'),str(PCA):sha(PCA)}
    assert dependencies[str(PCA)]==parent['dependencies'][str(PCA)]
    pack_files={};packs={};wanted=set()
    for panel,spec in parent['panels'].items():
        for task in spec['tasks']:
            with np.load(spec['cases'][task]['shortlist']) as z:
                wanted.update((task,int(i)) for i in z['action_idx'])
    for panel in ['cross_f1','within_f0']:
        for suffix in ['train_s16r16','train_clusters','dev']:
            prefix=D/'packs'/(panel+'_'+suffix+'_pca128');meta=Path(str(prefix)+'_meta.npz');latent=Path(str(prefix)+'_latents.npy')
            if not meta.exists() or not latent.exists():continue
            dependencies[str(meta)]=sha(meta);st=latent.stat();pack_files[str(latent)]={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
            with np.load(meta) as z:
                files=z['files'];tasks=z['tasks'];actions=z['action'];labels=z['solved']
            for i,(file,task) in enumerate(zip(files,tasks)):
                ai=int(Path(str(file)).stem.rsplit('_',1)[1]);key=(str(task),ai)
                if key in wanted and key not in packs:packs[key]={'kind':'pack','path':str(latent),'row':i,'action':actions[i].tolist(),'solved':bool(labels[i])}
    panels={}
    for panel,spec in parent['panels'].items():
        probe=P/'results'/('probes_pca128_cross/ball_cross_template_f1_full.pt' if panel=='cross' else 'probes_pca128/ball_within_template_f0_full.pt')
        head=P/'results/probe_adaptation_cross_20261001/heads/predicted_mix_s0/probe.pt' if panel=='cross' else probe
        dependencies[str(probe)]=sha(probe);dependencies[str(head)]=sha(head)
        for path in [probe,head]:assert dependencies[str(path)]==parent['dependencies'][str(path)]
        cases={};counts=Counter()
        for task in spec['tasks']:
            case=spec['cases'][task];path=Path(case['shortlist']);assert sha(path)==case['shortlist_sha256']
            labelpath=path.with_name(path.stem+'_labels.npy');assert sha(labelpath)==case['labels_sha256']
            with np.load(path) as z:indices=z['action_idx'];actions=z['action']
            labels=np.load(labelpath);sources=[]
            for ai,action,status in zip(indices,actions,labels):
                cached=OLD/'assets_fixed'/panel/'observations'/f'{task.replace(":","_")}_{int(ai):06d}.npz'
                if cached.exists():source={'kind':'observation','path':str(cached)}
                elif (task,int(ai)) in packs:
                    source=packs[(task,int(ai))]
                    np.testing.assert_array_equal(np.array(source['action'],np.float32),action)
                    assert source['solved']==bool(status>0)
                else:source={'kind':'encode'}
                counts[source['kind']]+=1;sources.append(source)
            cases[task]=dict(case,sources=sources)
        panels[panel]={'tasks':spec['tasks'],'cases':cases,'probe':str(probe),'head':str(head),'cache_counts':dict(counts)}
    for f in list((OLD/'code').glob('*.py'))+[OLD/'recovery_20261010/recovery_runner.py']:
        dependencies[str(f)]=sha(f)
    write(R/'PLAN.json',{'panels':panels,'dependencies':dependencies,'pack_files':pack_files,'pca':str(PCA),
        'scope':'True encoded future latent through unchanged frozen panel scorer; no predictor or training',
        'initial_frame':'Preserve exact common shortlist z0; replace future steps 1..31 only',
        'budgets':[1,3,5,9],'batch':32,'candidates':1024,'tasks_per_panel':100,
        'limits':'Diagnostic with privileged future observations; scorer-limited reference, not a mathematical success upper bound',
        'storage':'Only scores/receipts; discard newly simulated raw after encoding; leave old caches unchanged'})
    print('PREPARED',json.dumps({p:s['cache_counts'] for p,s in panels.items()}),flush=True)


class Observations:
    def __init__(self,panel,plan):
        self.panel=panel;self.plan=plan;self.worker=None;self.maps={}
        self.temp=R/'temporary'/f'{os.getpid()}';self.temp.mkdir(parents=True,exist_ok=True)
    def raw(self,task,ai,action,status):
        if self.worker is None:
            self.worker=subprocess.Popen([PHYRE,'-u',str(OLD/'code/data_cpu.py'),'serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        path=self.temp/f'{task.replace(":","_")}_{ai:06d}.npz'
        self.worker.stdin.write(json.dumps({'task':task,'action':action.tolist(),'out':str(path)})+'\n');self.worker.stdin.flush()
        while True:
            line=self.worker.stdout.readline()
            if not line:raise RuntimeError('CPU simulation worker exited')
            try:d=json.loads(line)
            except json.JSONDecodeError:continue
            assert d['ok'] and d['status']==int(status);return path
    def get(self,task,ai,action,status,source):
        if source['kind']=='observation':
            with np.load(source['path']) as z:
                assert str(z['task'])==task and int(z['status'])==int(status)
                assert str(z['pca_sha256'])==self.plan['dependencies'][str(PCA)]
                np.testing.assert_array_equal(z['action'],action);return z['latents'].copy(),None
        if source['kind']=='pack':
            path=source['path']
            if path not in self.maps:self.maps[path]=np.load(path,mmap_mode='r')
            return self.maps[path][source['row']].copy(),None
        return None,self.raw(task,ai,action,status)
    def close(self):
        if self.worker:
            self.worker.stdin.close();self.worker.wait(timeout=60)
        for path in self.temp.glob('*.npz'):path.unlink()
        self.temp.rmdir()


def runtime(panel,plan):
    sys.path.insert(0,str(OLD/'code'));sys.path.insert(0,str(OLD/'recovery_20261010'))
    import torch
    from common import AttnProbe
    import recovery_runner as m
    spec=plan['panels'][panel];qc=torch.load(spec['probe'],map_location='cpu',weights_only=False)
    hc=torch.load(spec['head'],map_location='cpu',weights_only=False)
    head=AttnProbe(hc['D'],hc['S']).cuda();head.load_state_dict(hc['state_dict']);head.eval().requires_grad_(False)
    torch.set_num_threads(4);return m,head,qc['rule']


def evaluate(panel,shard,nshards,smoke=False):
    import torch
    plan=checked();prov=provenance();spec=plan['panels'][panel]
    if smoke and (R/'preflight'/f'{panel}.json').exists():
        old=read(R/'preflight'/f'{panel}.json');assert old['passed'] and old['provenance']==prov
        return
    m,head,rule=runtime(panel,plan)
    from fast_inference import PreparedProbeReadout
    if not smoke:assert read(R/'preflight'/f'{panel}.json')['provenance']==prov
    obs=Observations(panel,plan);encoder=None;start=time.time();sample_times=[];validations=[]
    tasks=spec['tasks'][shard::nshards]
    if smoke:tasks=tasks[:1]
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            for task in tasks:
                folder=R/('smoke' if smoke else 'evaluation')/panel/task.replace(':','_');case=spec['cases'][task]
                path=Path(case['shortlist']);labelpath=path.with_name(path.stem+'_labels.npy')
                assert sha(path)==case['shortlist_sha256'] and sha(labelpath)==case['labels_sha256']
                with np.load(path) as z:actions=z['action'];indices=z['action_idx'];initial=z['z0'];features=z['features']
                labels=np.load(labelpath);sources=case['sources'];scores=np.full(1024,np.nan);completed=0
                if (folder/'scores.npz').exists():
                    assert read(folder/'receipt.json')['provenance']==prov
                    with np.load(folder/'scores.npz') as z:
                        np.testing.assert_array_equal(z['indices'],indices);np.testing.assert_array_equal(z['labels'],labels>0);scores=z['scores']
                    finite=np.isfinite(scores);completed=int(finite.sum());assert finite[:completed].all() and not finite[completed:].any()
                end=min(32,len(actions)) if smoke else len(actions)
                def submit(i):return pool.submit(obs.get,task,int(indices[i]),actions[i],int(labels[i]),sources[i])
                future=submit(completed) if completed<end else None;latents=[];chunk_start=completed
                for i in range(completed,end):
                    t0=time.time();truth,raw=future.result()
                    future=submit(i+1) if i+1<end else None
                    if raw is not None:
                        if encoder is None:encoder=m.Encoder(PCA)
                        with np.load(raw) as z:
                            assert str(z['task'])==task and int(z['status'])==int(labels[i]);np.testing.assert_array_equal(z['action'],actions[i])
                            truth=encoder.rollout(z['frames'])
                        raw.unlink()
                    assert truth.shape==(32,196,128) and np.isfinite(truth).all()
                    diff=np.abs(truth[0].astype(float)-initial[i]);assert diff.max()<.01 and diff.mean()<1e-6,(task,int(indices[i]),float(diff.max()),float(diff.mean()))
                    # Same exact known initial latent as every prediction method.
                    truth=truth.astype(np.float32);truth[0]=initial[i];latents.append(truth);sample_times.append(time.time()-t0)
                    if len(latents)==32 or i+1==end:
                        z=torch.from_numpy(np.stack(latents)).cuda();ff=[f[None] for f in features[chunk_start:i+1]]
                        values=PreparedProbeReadout(ff,32,z.device)(head,z,rule)
                        if chunk_start==0:
                            ref=m.p.score_function(head,rule)(z,ff).cpu().numpy();np.testing.assert_array_equal(ref,values)
                            validations.append({'task':task,'scorer_exact':True})
                        scores[chunk_start:i+1]=values
                        npz(folder/'scores.npz',scores=scores,indices=indices,labels=labels>0)
                        write(folder/'receipt.json',{'provenance':prov,'completed':i+1,'total':1024,'case':task,'initial_frame_canonical':True})
                        latents=[];chunk_start=i+1
                        write(R/'progress'/f'{panel}_{shard}_{int(smoke)}.json',{'task':task,'actions_done':i+1,'elapsed_s':time.time()-start,
                            'recent_action_mean_s':float(np.mean(sample_times[-128:])),'source_counts':dict(Counter(s['kind'] for s in sources[:i+1]))})
                if not smoke:
                    assert np.isfinite(scores).all();first=m.p.first_rank(scores,labels>0)
                    write(folder/'result.json',{'provenance':prov,'task':task,'first_success':first,'score_sha256':sha(folder/'scores.npz')})
                print('DONE',panel,task,'smoke',smoke,'seconds',time.time()-start,flush=True)
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
            obs.close()
    if smoke:
        write(R/'preflight'/f'{panel}.json',{'passed':True,'provenance':prov,'reference_scores_exact':validations,
            'actions':len(sample_times),'seconds':time.time()-start,'action_times_s':sample_times})
    else:write(R/'shards'/f'{panel}_{shard}.json',{'complete':True,'provenance':prov,'tasks':tasks,'seconds':time.time()-start})


def summary():
    plan=checked();prov=provenance();out={'complete':True,'provenance':prov,'panels':{}}
    for panel,spec in plan['panels'].items():
        rows=[]
        for task in spec['tasks']:
            folder=R/'evaluation'/panel/task.replace(':','_');row=read(folder/'result.json');assert row['provenance']==prov
            assert row['score_sha256']==sha(folder/'scores.npz')
            with np.load(folder/'scores.npz') as z:
                assert np.isfinite(z['scores']).all();order=np.argsort(-z['scores'],kind='stable');hits=np.flatnonzero(z['labels'][order]);first=int(hits[0]+1) if len(hits) else None
                with np.load(M/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as baseline:
                    np.testing.assert_array_equal(z['indices'],baseline['indices']);np.testing.assert_array_equal(z['labels'],baseline['labels'])
            assert first==row['first_success'];rows.append(row)
        out['panels'][panel]={'n':len(rows),'counts':{f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in rows) for k in [1,3,5,9]},'tasks':rows}
    write(R/'SUMMARY.json',out);print(json.dumps({p:d['counts'] for p,d in out['panels'].items()}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','smoke','eval','summary']);ap.add_argument('--panel',choices=['cross','within']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=3);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='summary':summary()
    else:evaluate(a.panel,a.shard,a.nshards,a.mode=='smoke')
