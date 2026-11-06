"""Independent NumPy audit; never imports training/evaluation implementations."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def metric(scores,labels,first=None):
 ids=[i for i in range(len(labels)) if first is None or i!=first];assert np.isfinite(scores).all()
 order=sorted(ids,key=lambda i:(-float(scores[i]),i));v=scores[ids];y=labels[ids];pos=v[y];neg=v[~y]
 auc=float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean()) if len(pos) and len(neg) else None
 return {**{f'top{k}':int(any(labels[i] for i in order[:k])) for k in [1,3,5,8]},'auc':auc}

def filescores(path):
 with np.load(path) as z:return {k:z[k].copy() for k in z.files}

def audit():
 plan=read(R/'PLAN.json');panels={};vectors=0;incomplete=[]
 for stage in ['discovery','confirmation']:
  panel={}
  for task in plan[stage]:
   name=task.replace(':','_');root=R/'evaluation/seed7919'/stage/name;p=root/'scores.npz'
   if not p.exists():incomplete.append(str(p));continue
   raw=filescores(p);first=int(raw['first']);labels=raw['labels'];failed=not bool(labels[first]);arrays={('q0' if k=='q0' else k+'_s7919'):v for k,v in raw.items() if k not in ['labels','indices','first']}
   r=read(root/'result.json');assert r['complete'] and r['first_failed']==failed and r['reader_unchanged'];assert first==int(np.argmax(raw['q0']))
   for k,expected in r['records'].items():
    actual=metric(raw[k],labels,first if failed else None)
    for field,val in expected.items():assert val is None and actual[field] is None or val is not None and abs(val-actual[field])<1e-12
   for receipt in [R/'evaluation/seed7920'/stage/name/'result.json',R/'calibration_update/evaluation'/stage/name/'result.json']:
    if not receipt.exists():incomplete.append(str(receipt))
    else:
     rr=read(receipt);assert rr['complete'] and rr['first_failed']==failed
   extras=[(R/'evaluation/seed7920'/stage/name/'scores.npz','seed'),(R/'calibration_update/evaluation'/stage/name/'scores.npz','cal'),(R/'readout/evaluation'/stage/name/'scores.npz','readout')]
   for path,kind in extras:
    if not failed:continue
    if not path.exists():incomplete.append(str(path));continue
    other=filescores(path)
    for key in ['indices','labels','first']:np.testing.assert_array_equal(other[key],raw[key])
    for k,v in other.items():
     if k in ['labels','indices','first']:continue
     if kind=='seed' and k=='q0':np.testing.assert_array_equal(v,raw['q0']);continue
     if kind=='readout' and k in ['original','scalar_calibration']:np.testing.assert_array_equal(v,raw['q0']);continue
     arrays[k+'_s7920' if kind=='seed' else k]=v
   metrics={k:metric(v,labels,first) for k,v in arrays.items()} if failed else {}
   vectors+=len(arrays);panel[task]=dict(failed=failed,metrics=metrics)
  panels[stage]=panel
 transfer_plan=R/'cross_transfer/PLAN.json'
 if transfer_plan.exists():
  panel={}
  extension=R/'cross_transfer/EXTENSION_PLAN.json'
  transfer_tasks=read(extension)['all_tasks'] if extension.exists() else read(transfer_plan)['tasks']
  for task in transfer_tasks:
   d=R/'cross_transfer/evaluation'/task.replace(':','_');p=d/'scores.npz'
   if not p.exists():incomplete.append(str(p));continue
   raw=filescores(p);r=read(d/'result.json');first=int(raw['first']);labels=raw['labels'];failed=not bool(labels[first]);assert r['complete'] and r['reader_unchanged'] and r['first_failed']==failed and first==int(np.argmax(raw['q0']))
   arrays={k:v for k,v in raw.items() if k not in ['labels','indices','first']};metrics={k:metric(v,labels,first) for k,v in arrays.items()} if failed else {}
   static={k:metric(arrays[k],labels) for k in ['q0','classifier','query8_readout','fullpool_readout']}
   vectors+=len(arrays);panel[task]=dict(failed=failed,metrics=metrics,static=static)
  panels['cross20']={t:r for t,r in panel.items() if t in read(transfer_plan)['tasks']}
  if extension.exists():panels['cross100']=panel
 result={}
 for stage,panel in panels.items():
  if not panel:continue
  active=[t for t,r in panel.items() if r['failed']];names=sorted(set.intersection(*(set(panel[t]['metrics']) for t in active))) if active else []
  n=len(panel);fail=len(active);summary={}
  for name in names:
   ms=[panel[t]['metrics'][name] for t in active]
   summary[name]={k:float(np.mean([m[k] for m in ms if m[k] is not None])) if any(m[k] is not None for m in ms) else None for k in ['top1','top3','top5','top8','auc']}
   summary[name]['counts']={k:sum(m[k] for m in ms) for k in ['top1','top3','top5','top8']}
   summary[name]['two_attempt_success_count']=n-fail+sum(m['top1'] for m in ms)
   summary[name]['first_plus_next8_static_success_count']=n-fail+sum(m['top8'] for m in ms)
  pairs={}
  comparisons=[]
  for seed in [7919,7920]:
   for loss in ['bce','pair']:
    a=f'{loss}_residual_s{seed}'
    for b in [f'prediction_residual_s{seed}',f'{loss}_zero_s{seed}','q0']:
     if a in names and b in names:comparisons.append((a,b))
  for a,b in [('bcecal_residual','bce_residual_s7919'),('bcecal_residual','bcecal_zero'),('query8_readout','q0'),('fullpool_readout','q0')]:
   if a in names and b in names:comparisons.append((a,b))
  for a,b in comparisons:
   values={k:np.array([panel[t]['metrics'][a][k]-panel[t]['metrics'][b][k] for t in active]) for k in ['top1','top8']}
   pairs[a+' vs '+b]={k:dict(gain=int((v>0).sum()),loss=int((v<0).sum()),tie=int((v==0).sum()),net=int(v.sum())) for k,v in values.items()}
  entry=dict(n=n,first_failed=fail,first_success=n-fail,summary=summary,paired=pairs,per_scene=panel)
  if stage.startswith('cross'):
   entry['static_own_first']={a:{k:float(np.mean([r['static'][a][k] for r in panel.values() if r['static'][a][k] is not None])) for k in ['top1','top3','top5','top8','auc']} for a in ['q0','classifier','query8_readout','fullpool_readout']}
  result[stage]=entry
 training={}
 for seed in [7919,7920]:
  rows=[read(p) for p in (R/'models'/f'seed{seed}').glob('*/complete.json')]
  if not rows:continue
  for k in ['initial_sha','reader_sha','query_schedule_sha']:assert len({r[k] for r in rows})==1,(seed,k)
  assert len(rows)==(6 if seed==7919 else 4),(seed,len(rows))
  training[str(seed)]=dict(n=len(rows),steps=[r['steps'] for r in rows],pair_steps=[r['active_pair_steps'] for r in rows],initial_sha=rows[0]['initial_sha'],reader_sha=rows[0]['reader_sha'])
 if len(training)==2:assert training['7919']['reader_sha']==training['7920']['reader_sha'] and training['7919']['initial_sha']!=training['7920']['initial_sha']
 calrows=[read(p) for p in (R/'calibration_update/models').glob('*/complete.json')]
 assert len(calrows)==2
 for r in calrows:
  assert r['initial_sha']==training['7919']['initial_sha'] and r['reader_sha']==training['7919']['reader_sha']
 manifest=[]
 for p in sorted(R.rglob('scores.npz')):manifest.append(str(p.relative_to(R))+' '+hashlib.sha256(p.read_bytes()).hexdigest())
 (R/'RAW_SCORES_SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n')
 report=dict(complete=not incomplete,raw_vectors_recomputed=vectors,incomplete=incomplete,training=training,panels=result,score_manifest_sha256=hashlib.sha256('\n'.join(manifest).encode()).hexdigest(),scope='Independent raw scores and labels, stable Python sorting, direct pairwise AUC; paired common-history comparisons. Geometry losses not independently recomputed. Cross20 two-attempt counts reconstruct firstQ0 trial then one update using cached simulation labels. next8 remains static, not repeated-feedback S9.')
 (R/'INDEPENDENT_ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE',not incomplete,'VECTORS',vectors,'MISSING',len(incomplete))
 for stage,p in result.items():
  print(stage,'N',p['n'],'firstfails',p['first_failed'])
  for name,x in p['summary'].items():print(name,x['counts'],'S2',x['two_attempt_success_count'],'auc',x['auc'])

if __name__=='__main__':audit()
