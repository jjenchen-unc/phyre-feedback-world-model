"""Oracle partial correction of goal surface-distance trajectory, no fitting.

Future query truth is privileged diagnostic information, not deploy feedback.
Preserve current Q0/pool; use archived true distances only after exact identity checks.
"""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;P=R.parents[1];O=P/'results/oracle_small_1024_20261017'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
betas=[0.,.1,.25,.5,1.]
plan=dict(source_sha=sha(Path(__file__)),oracle_summary_sha=sha(O/'SUMMARY.json'),betas=betas,scope='Ten historically oracle-encoded cross scenes, matched CURRENT Q0 predictions/1024pool. Interpolate surface-distance trajectory toward simulatortruth; no training, no latent interpolation or real support feedback. Tests target relevance, not learnability or a deployablemethod.',selection='All10crossscenes in prior oracle summary; no selection on currentmethodoutcomes',interventions=['full_distance_residual','time_mean_only_residual'],metric='Same commonfailedfirstQ0 masked; top1/3/5/8 andAUC; oraclefulltrajectoryerror reductions measured onidenticalqueries.')
assert not(R/'CONTACT_TARGET_PLAN.json').exists();(R/'CONTACT_TARGET_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
def score(d):
 margin=4-np.asarray(d,float);return np.logaddexp.reduce(-np.logaddexp(0.,-margin),axis=-1)
def metrics(s,y,first):
 ids=[i for i in range(len(s)) if i!=first];order=sorted(ids,key=lambda i:(-float(s[i]),i));v=s[ids];yy=y[ids];pos=v[yy];neg=v[~yy]
 return {**{f'top{k}':int(any(y[i] for i in order[:k])) for k in [1,3,5,8]},'auc':float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean()) if len(pos) and len(neg) else None}
rows={};checks=[]
for old in read(O/'SUMMARY.json')['panels']['cross']['tasks']:
 task=old['task'];op=O/'tasks/cross'/task.replace(':','_')/'scores.npz';assert sha(op)==old['scores_sha256']
 with np.load(op) as z:truth={k:z[k].copy() for k in z.files}
 cp=R/'cross_transfer/data'/task.replace(':','_')/'q0.npz'
 with np.load(cp) as z:now={k:z[k].copy() for k in z.files}
 for key in ['indices','labels']:np.testing.assert_array_equal(now[key],truth[key])
 np.testing.assert_array_equal(now['scores'],truth['q0']);first=int(now['first']);assert not now['labels'][first]
 d=now['distance'].astype(float);td=truth['true_distance'].astype(float);np.testing.assert_allclose(score(d),now['scores'],rtol=0,atol=5e-4)
 np.testing.assert_allclose(score(td),truth['oracle_geometry'],rtol=0,atol=1e-12)
 checks.append(dict(task=task,oracle_sha=sha(op),current_sha=sha(cp),current_q0_exact=True,oracle_geometry_recomputed=True))
 row={};residual=td-d
 for kind,delta in [('full_distance_residual',residual),('time_mean_only_residual',residual.mean(1,keepdims=True))]:
  for beta in betas:
   adjusted=d+beta*delta;value=now['scores'] if beta==0 else score(adjusted)
   row[f'{kind}_b{beta}']=dict(**metrics(value,now['labels'],first),distance_mae=float(abs(adjusted-td).mean()))
 rows[task]=row
names=list(next(iter(rows.values())))
summary={name:{k:float(np.mean([row[name][k] for row in rows.values()])) for k in ['top1','top3','top5','top8','auc','distance_mae']} for name in names}
out=dict(complete=True,plan_sha=sha(R/'CONTACT_TARGET_PLAN.json'),n=len(rows),checks=checks,summary=summary,per_scene=rows,interpretation='Oracle query-specific correction is an idealized intervention; even a positive result cannot prove a support-driven updater can infer these corrections. Time-mean-only correction is a distance-trajectory control, NOT the earlier latentresidual-centering experiment.')
(R/'CONTACT_TARGET_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
