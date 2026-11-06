"""Independent NumPy/SciPy replay of neural readout on deterministic scenes."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import ndtr
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def features(z,pc):
    j=int(z['first']);n=len(z['q']);x=np.empty((n,131),dtype=np.float64)
    x[:,:37]=z['q'];x[:,37:40]=z['actions'];x[:,40:77]=z['q'][j];x[:,77:80]=z['actions'][j]
    x[:,80:112]=(z['distance']-z['distance'][j])/256;x[:,112:115]=z['actions']-z['actions'][j];x[:,115:]=pc
    return x
p=read(R/'PLAN.json');meta=read(R/'DATA.json');norm=read(R/'NORMALIZATION.json');fit=[]
for task in p['panels']['fit']:
    if not meta['records'][task]['failed']:continue
    z=load(R/'data'/task.replace(':','_')/'features.npz');fit.append(features(z,z['residual_pc'])[z['query']])
x=np.concatenate(fit);mean=x.mean(0);mean[115:]=0;std=np.maximum(x.std(0),1e-5)
np.testing.assert_array_equal(mean,norm['mean']);np.testing.assert_array_equal(std,norm['std'])
selected={}
for stage in ['discovery','confirmation','cross100']:
    groups={}
    for task in sorted(p['panels'][stage]):
        if meta['records'][task]['failed']:groups.setdefault(task[:5],task)
    selected[stage]=list(groups.values())
records=[]
for seed in p['seeds']:
 for arm in p['arms']:
    d=R/'runs'/f'{seed}_{arm}';w=load(d/'weights.npz')
    for stage,tasks in selected.items():
     for task in tasks:
        name=task.replace(':','_');z=load(R/'data'/name/'features.npz');saved=load(d/'evaluation'/stage/name/'scores.npz')
        for mode in ['actual']+(['shuffle0'] if arm=='real' else []):
            pc=z['residual_pc']
            if mode=='shuffle0':pc=load(R/'data'/meta['shuffle'][stage][0][task].replace(':','_')/'features.npz')['residual_pc']
            v=np.clip((features(z,pc)-mean)/std,-10,10).astype(np.float32)
            if arm=='zero':v[:,115:]=0
            if arm=='query_only':v[:,40:]=0
            for layer in ['0','2']:
                v=v@w[layer+'.weight'].T+w[layer+'.bias'];v=v*ndtr(v)
            scores=z['baseline']+(v@w['4.weight'].T+w['4.bias']).ravel();err=float(abs(scores-saved[mode]).max());assert err<1e-4,(task,seed,arm,err)
            for k in [1,3,5,8]:
                aa=scores.copy();bb=saved[mode].copy();aa[int(z['first'])]=bb[int(z['first'])]=-np.inf
                assert bool(z['labels'][np.argsort(-aa,kind='stable')[:k]].any())==bool(z['labels'][np.argsort(-bb,kind='stable')[:k]].any()),(task,seed,arm,k)
            records.append(dict(task=task,seed=seed,arm=arm,mode=mode,max_error=err))
hashes=read(R/'REMOTE_HASHES.json')
for path,h in hashes.items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h,path
report=dict(complete=True,train_only_normalization_exact=True,remote_local_files_matched=len(hashes),replayed_vectors=len(records),max_error=max(v['max_error'] for v in records),topk_success_agreement=True,selected_scenes=selected,records=records,
    scope='Independent NumPy featureassembly/normalization and SciPy GELU neuralforward. Deterministic firstactive scene pertemplate/stage; all1024candidates. Topk successesagree; floatroundoffallowed inlogits. Allrawfiles hashchecked.')
(R/'REPLAY_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='records'})
