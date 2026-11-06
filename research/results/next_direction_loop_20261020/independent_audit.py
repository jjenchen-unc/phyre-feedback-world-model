import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/path/to/phyre-unc/results/next_direction_loop_20261020')
plan=json.loads((R/'PLAN.json').read_text()); final=json.loads((R/'FINAL_SUMMARY.json').read_text()); ph=hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest()
def metric(scores,labels,tried):
 ids=[i for i in range(len(labels)) if i not in set(map(int,tried))]
 assert all(np.isfinite(scores[ids]))
 order=sorted(ids,key=lambda i:(-float(scores[i]),i))
 y=np.asarray(labels)[ids];v=np.asarray(scores)[ids]
 pos=v[y];neg=v[~y]
 auc=float(np.mean((pos[:,None]>neg[None,:])+0.5*(pos[:,None]==neg[None,:]))) if len(pos) and len(neg) else None
 return {**{f'top{k}':int(any(labels[i] for i in order[:k])) for k in (1,3,5,8)},'auc':auc,'positive':int(y.sum()),'available':len(ids)}
def same(a,b):
 assert a.keys()==b.keys(),(a.keys(),b.keys())
 for k in a:
  assert a[k] is None and b[k] is None or a[k] is not None and b[k] is not None and abs(a[k]-b[k])<1e-12,(k,a[k],b[k])
count=0;stage_out={}
for stage,report in [('discovery',final['discovery']['report']),('confirmation',final['confirmation'])]:
 allrows={}; failures=[];pairs={}; arrays={}
 for task in plan[stage]:
  d=R/stage/task.replace(':','_'); rec=json.loads((d/'result.json').read_text())
  assert rec['complete'] and rec['old_exact'] and rec['model_unchanged'] and rec['plan']==ph
  arrays[task]={}
  for row in rec['records']:
   variant=row['variant'];z=np.load(d/(variant+'.npz'));arrays[task][variant]=z
   if variant=='q0':
    assert len(z['tried'])==0
    assert int(z['qfirst'])==int(np.argmax(z['old']))
    assert rec['first_failed']==(not bool(z['labels'][int(z['qfirst'])]))
   else: assert z['tried'].tolist()==[int(z['qfirst'])]
   for name,h in row['heads'].items():
    m=metric(z[name],z['labels'],z['tried']);same(m,h['rank']);count+=1
    if rec['first_failed']:same(metric(z[name],z['labels'],[int(z['qfirst'])]),h['after_first_failure'])
   allrows.setdefault(variant,[]).append(row)
  if rec['first_failed']:
   failures.append(task);q=arrays[task]['q0']
   for variant,z in arrays[task].items():
    assert np.array_equal(z['indices'],q['indices']) and np.array_equal(z['labels'],q['labels']) and z['qfirst']==q['qfirst']
 assert len(failures)==report['eligible_first_failures']
 for variant,rows in allrows.items():
  for name in rows[0]['heads']:
   avg={k:float(np.mean([row['heads'][name]['rank'][k] for row in rows if row['heads'][name]['rank'][k] is not None])) if any(row['heads'][name]['rank'][k] is not None for row in rows) else None for k in ['top1','top3','top5','top8','auc']}
   same(avg,report['summary'][variant]['heads'][name]['rank'])
 for name,t in report['transfer'].items():
  qname='q0_mixed_'+name.rsplit('_',1)[1] if name not in ['old','previous_mixed'] else name
  pairs[name]={}
  for comparison,variant,head in [('zero','smooth_zero',name),('q0_old','q0','old'),('q0_domain','q0',qname)]:
   diffs={k:[] for k in ('top1','top8')};vals=[];mem=[]
   for task in failures:
    az=arrays[task];z=az[variant];m=az['smooth_centered']
    mv=metric(m[name],m['labels'],[int(m['qfirst'])]);cv=metric(z[head],z['labels'],[int(z['qfirst'])]);vals.append(cv);mem.append(mv)
    for k in diffs:diffs[k].append(mv[k]-cv[k])
   for key,rows in [(comparison,vals),('memory',mem)]:
    avg={k:float(np.mean([row[k] for row in rows if row[k] is not None])) if any(row[k] is not None for row in rows) else None for k in ['top1','top3','top5','top8','auc']};same(avg,t[key])
   pairs[name][comparison]={k:{'gain':sum(x>0 for x in ds),'loss':sum(x<0 for x in ds),'tie':sum(x==0 for x in ds)} for k,ds in diffs.items()}
 stage_out[stage]={'first_failure_n':len(failures),'paired_counts':pairs}
files=sorted(p for p in R.rglob('*') if p.is_file() and (p.parent.parent.name in ['discovery','confirmation']) and p.suffix in ['.json','.npz'])
manifest='\n'.join(str(p.relative_to(R))+' '+hashlib.sha256(p.read_bytes()).hexdigest() for p in files)
out={'passed':True,'ranking_vectors_recomputed':count,'method':'Independent Python stable sorting and pairwise positive-negative AUC; record/summary/transfer checks; same pool, labels, first action, trial masking checked. Geometry summaries not independently recomputed from latent caches.','plan_sha256':ph,'final_sha256':hashlib.sha256((R/'FINAL_SUMMARY.json').read_bytes()).hexdigest(),'raw_files':len(files),'raw_manifest_sha256':hashlib.sha256(manifest.encode()).hexdigest(),'stages':stage_out}
(R/'COMPLETION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');(R/'RAW_SHA256SUMS.txt').write_text(manifest+'\n')
print({k:v for k,v in out.items() if k!='stages'})
for stage,r in stage_out.items():
 print(stage)
 for name,p in r['paired_counts'].items():
  if name.startswith('memory_predicted'):print(name,p)
