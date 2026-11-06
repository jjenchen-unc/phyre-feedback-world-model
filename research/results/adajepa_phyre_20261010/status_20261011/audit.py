import json,hashlib
from pathlib import Path
import numpy as np
P=Path('/path/to/phyre-unc');M=P/'results/small_official_20261010';R=P/'results/adajepa_phyre_20261010/status_20261011'
def read(p):return json.loads(p.read_text())
selection=read(M/'selection.json')['panels'];output={};original={};cases=0
for panel in ['within','cross']:
 output[panel]={}
 for folder in sorted((M/'evaluation_fixed'/panel).iterdir()):
  if not (folder/'complete.json').exists():continue
  done=read(folder/'complete.json');assert done['n']==100
  rows=[];isq=folder.name.startswith('q0_');seed=folder.name.rsplit('_s',1)[1]
  for task in selection[panel]['selected']:
   sub=folder/task.replace(':','_');r=read(sub/'result.json');z=np.load(sub/'scores.npz')
   pool=M/'assets_fixed'/panel/'shortlist'/(task.replace(':','_')+'.npz');d=np.load(pool);status=np.load(pool.with_name(pool.stem+'_labels.npy'))
   np.testing.assert_array_equal(z['indices'],d['action_idx']);np.testing.assert_array_equal(z['labels'],status>0)
   assert len(set(z['indices'].tolist()))==1024
   labels=z['labels']
   if isq:
    for key,field in [('q0','first_success'),('original','original_first_success')]:
     assert np.isfinite(z[key]).all();hits=np.flatnonzero(labels[np.argsort(-z[key],kind='stable')]);hit=int(hits[0])+1 if len(hits) else None;assert hit==r[field]
    k=(panel,task)
    if k in original:np.testing.assert_array_equal(z['original'],original[k])
    else:original[k]=z['original'].copy()
   else:
    tried=set();hit=None
    for i,(scores,j) in enumerate(zip(z['scores'],z['selected']),1):
     assert np.isfinite(scores).all();s=scores.copy();s[list(tried)]=-np.inf
     assert int(np.argmax(s))==j and j not in tried;tried.add(int(j))
     if labels[j]:assert i==len(z['selected']);hit=i
    assert hit==r['first_success'] and len(tried)==r['attempts']<=20 and r['updates']==r['attempts']-1
    with np.load(M/'evaluation_fixed'/panel/('q0_s'+seed)/task.replace(':','_')/'scores.npz') as q:np.testing.assert_array_equal(q['q0'],z['scores'][0])
   rows.append(r);cases+=1
  counts={f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in rows) for k in [1,3,5,9,20]};assert counts==done['counts']
  output[panel][folder.name]={'counts':counts,'first_success':{r['task']:r['first_success'] for r in rows}}
  if isq:output[panel]['original']={'counts':{f's{k}':sum(r['original_first_success'] is not None and r['original_first_success']<=k for r in rows) for k in [1,3,5,9,20]}}
for arm in ['q0','residual','zero']:
 output['within'][arm+'_mean']={f's{k}':float(np.mean([output['within'][f'{arm}_s{s}']['counts'][f's{k}'] for s in range(3)])) for k in [1,3,5,9,20]}
for seed in range(3):
 output['within'][f'paired_s{seed}']={}
 for other in ['q0','zero']:
  a=output['within'][f'residual_s{seed}']['first_success'];b=output['within'][f'{other}_s{seed}']['first_success'];wins=losses=0
  for t in a:
   av=a[t] is not None and a[t]<=9;bv=b[t] is not None and b[t]<=9;wins+=av and not bv;losses+=bv and not av
  output['within'][f'paired_s{seed}'][other]={'gained':wins,'lost':losses}
R.mkdir(exist_ok=True);result={'raw_audit_passed':True,'complete_cases':cases,'splits':output};(R/'RESULTS.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'cases':cases,'counts':{p:{k:v.get('counts',v) for k,v in d.items() if 'first_success' not in v or 'counts' in v} for p,d in output.items()}},indent=2))
