"""Verify training positive coverage and paired zero-residual fixed-query control."""
import json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;R=HERE.parent
read=lambda p:json.loads(p.read_text())
m=read(R/'manifest.json');files=[];labels=[];seen=set()
for name in ['cross_f1_train_clusters_pca128','cross_f1_train_s16r16_pca128']:
    with np.load('/mnt/meg/jjenchen/phyre-unc/packs/'+name+'_meta.npz') as d:
        for file,solved in zip(d['files'],d['solved']):
            file=str(file)
            if file in seen:continue
            seen.add(file);files.append(file);labels.append(bool(solved))
assert hashlib.sha256('\n'.join(files).encode()).hexdigest()==m['store_files_sha256']
y=np.array(labels);rr=[r for r in m['rows'] if r['split']=='train']
n=sum(len(r['query']) for r in rr);positive=sum(int(y[r['query']].sum()) for r in rr)
assert n==m['counts']['train']['queries'] and positive==m['counts']['train']['positive_queries']
out={'training':{'n_scenes':len(rr),'unique_query_count':n,'positive_query_count':positive,
                 'positive_fraction':positive/n,'expected_equal_scene_positive_fraction':float(np.mean([y[r['query']].mean() for r in rr])),
                 'scope':'Registered query pools; expected objective weight for uniform scene/query sampling, not actual epoch draw counts'},'zero_controls':{}}
for template in ['00000','00002','00007','00014']:
    out['zero_controls'][template]={}
    for arm in ['residual','zero']:
        v=[]
        for p in (R/'evaluation').glob(arm+'_s*/fixed/'+template+'*/summary.json'):
            d=read(p);b=d['conditions']['predicted_mix_s0__base'];u=d['conditions']['predicted_mix_s0__memory']
            if not d['context'] or not b['n_success']:continue
            v.append((b['auc'],u['auc'],b['success']['surface_mae'],u['success']['surface_mae']))
        out['zero_controls'][template][arm]=dict(n=len(v),**dict(zip(['q0_auc','updated_auc','q0_success_geometry','updated_success_geometry'],np.mean(v,axis=0).tolist())))
(HERE/'CONTROLS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
