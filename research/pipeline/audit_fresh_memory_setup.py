"""Independent engineering check before launching fresh training; no effect gate."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'code'))
import fresh_memory_geometry as f
from redesign_lib import Store
m=f.read(root/'manifest.json');assert 'memory' not in m and 'fresh' in m['scope'].lower()
store=Store(include_features=False);rows=[r for r in m['rows'] if r['split']=='train']
counts={}
for seed in m['seeds']:
    scenes=set();pos=total=0
    for step in range(m['training']['steps']):
        r,q,times,ctx=f.episode(rows,seed,step)
        assert not set(q)&set(ctx) and not store.labels[ctx].any()
        assert len(q)==len(set(q))==min(8,len(r['query']))
        assert set(q)<=set(r['query']) and set(ctx)<=set(r['failures'])
        assert all(str(store.tasks[i])==r['task'] for i in q+ctx)
        scenes.add(r['task']);pos+=int(store.labels[q].sum());total+=len(q)
    counts[str(seed)]={'scheduled_scenes_seen':len(scenes),'scheduled_query_count':total,'scheduled_positive_query_fraction':pos/total}
    assert len(scenes)==1200
smoke=root/'smoke/seed_0';done=f.read(smoke/'complete.json');ck=Path(done['checkpoint'])
assert f.digest(ck)==done['checkpoint_sha256']
a=torch.load(smoke/'checkpoints/step_000000.pt',map_location='cpu',weights_only=False)
b=torch.load(ck,map_location='cpu',weights_only=False)
assert all(not k.startswith('base.') for k in a['state_dict'])
assert not torch.equal(a['state_dict']['zin.1.weight'],b['state_dict']['zin.1.weight'])
base,probe,rule,model,heads=f.models(m,0,ck);cross=f.Pack(m['cross_pack'])
# A mixed-label case ensures numerical AUC and ranking branches execute; only engineering selection.
spec=next(r for r in m['final_panels'] if r['panel']=='cross_natural' and r['context'] and cross.labels[r['query']].any() and not cross.labels[r['query']].all())
prov={'manifest':f.digest(root/'manifest.json'),'checkpoint':f.digest(ck),'auditor':f.digest(__file__)}
folder=root/'full_pool_smoke'/spec['task'].replace(':','_')
f.evaluate_panel(folder,spec,cross,base,model,heads,rule,prov,replay=f.PROBES/'evaluation/cross'/spec['task'].replace(':','_'))
r,n=f.audit_panel(folder)
for key,metric in r['conditions'].items():
    assert metric['auc'] is not None
    with np.load(folder/'truth.npz') as tt, np.load(folder/(key+'.npz')) as vv:
        mask=tt['query_mask'];y=tt['labels'][mask];s=vv['log_score'][mask]
    successes=np.sort(s[y]);failures=np.sort(s[~y]);lower=np.searchsorted(failures,successes,side='left');upper=np.searchsorted(failures,successes,side='right')
    auc=float((lower+.5*(upper-lower)).sum()/(len(successes)*len(failures)))
    assert abs(auc-metric['auc'])<1e-12
out={'passed':True,'manifest_sha256':f.digest(root/'manifest.json'),'code':f.code_hashes(),'auditor_sha256':f.digest(__file__),
     'full_pool_task':spec['task'],'full_pool_arrays_verified':n,'scheduled_draws_checked':60000,'scheduled_counts':counts,
     'checks':['fresh checkpoint reload','memory encoder actually changed','no base weights in memory checkpoint','all1200scenes sampled each seed',
     'allhead full1024base replay','numerical AUC independently recomputed by searchsorted','no performance acceptance criterion']}
f.write_json(root/'GPU_SMOKE_AUDIT.json',out);print(json.dumps(out,indent=2),flush=True)
