import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/path/to/phyre-unc/results/milestone_memory_20261015');D=Path('/mnt/meg/jjenchen/phyre-unc/packs')
p=json.loads((R/'PLAN.json').read_text());diag=json.loads((R/'DIAGNOSTICS.json').read_text());out={}
for panel,fold in [('cross','cross_f1'),('within','within_f0')]:
 files=[];labels=[];seen=set()
 for name in [fold+'_train_clusters_pca128',fold+'_train_s16r16_pca128']:
  with np.load(D/(name+'_meta.npz')) as a:fs=a['files'];ys=a['solved']
  for f,y in zip(fs,ys):
   if str(f) not in seen:seen.add(str(f));files.append(str(f));labels.append(bool(y))
 assert hashlib.sha256('\n'.join(files).encode()).hexdigest()==p['panels'][panel]['store_files_sha256']
 labels=np.array(labels);m=json.loads(Path(p['panels'][panel]['source_manifest']).read_text());rows=[x for x in m['rows'] if x['split']=='train']
 pool={}
 for group in ['candidates','query']:
  ys=[labels[row[group]] for row in rows]
  pool[group]={'scenes':len(ys),'count_median':float(np.median([len(y) for y in ys])),'count_min':min(map(len,ys)),'count_max':max(map(len,ys)),'success_fraction_mean_per_scene':float(np.mean([y.mean() for y in ys])),'success_fraction_median_per_scene':float(np.median([y.mean() for y in ys])),'scenes_with_success':sum(bool(y.any()) for y in ys),'all_success_scenes':sum(bool(y.all()) for y in ys)}
 ys=[];has={}
 for f in (R/'evaluation'/panel/'q0').glob('*/scores.npz'):
  with np.load(f) as a:ys.append(a['labels']);has[f.parent.name]=bool(a['labels'].any())
 pool['eval']={'count':1024,'success_fraction_mean_per_scene':float(np.mean([y.mean() for y in ys])),'success_fraction_median_per_scene':float(np.median([y.mean() for y in ys])),'scenes_with_success':sum(bool(y.any()) for y in ys)}
 pool['real_eval_solvable_reach_depth']={k:sum(has[f.parent.name] and len(json.loads(f.read_text())['events'])>k for f in (R/'evaluation'/panel/'residual').glob('*/result.json')) for k in [1,2,4,8]}
 out[panel]=pool;print(panel,json.dumps(pool))
(R/'POOL_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2)+'\n')
