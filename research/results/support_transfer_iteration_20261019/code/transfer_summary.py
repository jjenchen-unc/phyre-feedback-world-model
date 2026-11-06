from transfer_core import *
from transfer_probe import rho
import argparse,numpy as np

def decision():
 plan=checked();out={'complete':True,'panels':{},'plan_sha256':sha(R/'PLAN.json')}
 for panel in ['cross','within']:
  assert read(R/f'SMOKE_{panel}.json')['passed'];rows=[read(p) for p in sorted((R/'first'/panel).glob('*/*/result.json'))];assert len(rows)==16
  analysis={};distances=[]
  for split in ['train','evaluation']:
   part=[x for x in rows if x['split']==split];assert len(part)==8;correlations={'action':[],'motion':[],'oracle':[]};counts={'wrong_direction':0,'overshoot':0,'not_harmful':0};query_n=0
   for r in part:
    q=r['queries'];gain=[x['mse_gain'] for x in r['methods']['raw']['queries']]
    for key,x in [('action',[-x['action_distance'] for x in q]),('motion',[-x['predicted_motion_distance'] for x in q]),('oracle',[x['support_query_residual_cosine'] for x in q])]:
     value=rho(x,gain)
     if value is not None:correlations[key].append(value)
    for x in r['methods']['raw']['queries']:counts[x['harm_type']]+=1;query_n+=1
    if split=='train':distances.extend(x['action_distance'] for x in q)
   analysis[split]={'scenes':len(part),'query_count':query_n,'raw_correction_harm_counts':counts,'median_within_scene_spearman':{k:float(np.median(v)) if v else None for k,v in correlations.items()},'positive_correlation_scenes':{k:sum(x>0 for x in v) for k,v in correlations.items()},'ranking':{m:{'next_successes':sum(r['methods'][m]['ranking']['next_success'] for r in part),'top8_successes':sum(r['methods'][m]['ranking']['top8_success'] for r in part)} for m in ['raw','centered','norm_matched']}}
  a=analysis['train']['median_within_scene_spearman']['action'];b=analysis['evaluation']['median_within_scene_spearman']['action'];gate=a is not None and b is not None and a>0 and b>0 and analysis['train']['positive_correlation_scenes']['action']>=5 and analysis['evaluation']['positive_correlation_scenes']['action']>=5
  histories=[read(p) for p in sorted((R/'history'/panel).glob('*/result.json'))];assert len(histories)==8
  hs={}
  for origin in ['raw','centered']:
   for op in ['raw','centered','norm_matched']:
    rec=[next(r for r in h['records'] if r['history']==origin and r['operator']==op and r['updates']==h['depth']) for h in histories if h['depth']>1]
    hs[origin+'_'+op]={'n_later_states':len(rec),'next_successes':sum(r['metrics']['next_success'] for r in rec),'top8_successes':sum(r['metrics']['top8_success'] for r in rec),'mean_l1':float(np.mean([r['metrics']['l1'] for r in rec])) if rec else None,'mean_geometry':float(np.mean([r['metrics']['geometry'] for r in rec])) if rec else None}
  out['panels'][panel]={'first_support':analysis,'history':hs,'run_locality_gate':gate,'action_sigma':float(np.median(distances)),'decision':'Always validate norm-matched fullpolicy; additionally test smooth action-distance gate with uniform/shuffled controls only when association direction repeats in both diagnostic splits. This is screening, not significance.'}
 write(R/'ROUND2_DECISION.json',out);print(json.dumps(out,indent=2))

def collect(panel,policy):
 plan=checked();first={};checkpoint=read(B/'models'/panel/'uniform/checkpoint_128.json')['checkpoint_sha256']
 for task in plan['tasks'][panel]:
  path=R/'policies'/panel/policy/task.replace(':','_');row=read(path/'result.json');assert row['complete'] and row['first_scores_exact'] and row['model_unchanged'] and row['plan_sha256']==sha(R/'PLAN.json') and row['checkpoint_sha256']==checkpoint
  with np.load(path/'scores.npz') as z,np.load(S/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as ref:
   np.testing.assert_array_equal(z['indices'],ref['indices']);np.testing.assert_array_equal(z['labels'],ref['labels']);np.testing.assert_array_equal(z['scores'][0],ref['scores'][0]);sel=z['selected'];assert len(sel)==len(set(sel))==len(z['scores']) and np.isfinite(z['scores']).all()
   for k,j in enumerate(sel):
    v=z['scores'][k].copy();v[sel[:k]]=-np.inf;assert int(v.argmax())==j
   hits=np.flatnonzero(z['labels'][sel]);actual=int(hits[0]+1) if len(hits) else None;assert actual==row['first_success'] and len(sel)==(actual or 9)
   if policy!='norm_matched_full' and len(sel)>2:
    for v in z['scores'][2:]:np.testing.assert_array_equal(v,z['scores'][1])
  first[task]=actual
 return metrics(first)

def final():
 plan=checked();freeze=read(R/'ROUND0_FREEZE.json');d=read(R/'ROUND2_DECISION.json');out={'complete':True,'plan_sha256':sha(R/'PLAN.json'),'panels':{},'limits':plan['limits']};lines=['# Support-to-query transfer investigation','','All interventions use the frozen raw-trained128 updater, common Q0 first action and classifier1024 pool. No new updater training.','']
 for panel in ['cross','within']:
  methods=dict(freeze['panels'][panel]['methods']);methods['norm_matched_full']=collect(panel,'norm_matched_full');comparisons={'norm_vs_raw':paired(methods['norm_matched_full'],methods['raw_full']),'norm_vs_centered':paired(methods['norm_matched_full'],methods['centered_full'])}
  if d['panels'][panel]['run_locality_gate']:
   for policy in ['action_gate_first','shuffled_gate_first','uniform_gate_first']:methods[policy]=collect(panel,policy)
   for baseline in ['raw_first_only','shuffled_gate_first','uniform_gate_first']:comparisons['action_gate_vs_'+baseline]=paired(methods['action_gate_first'],methods[baseline])
  statements=[];n=methods['norm_matched_full']['counts']['s9'];raw=methods['raw_full']['counts']['s9'];center=methods['centered_full']['counts']['s9']
  statements.append('Norm-matched raw-direction updates reached or exceeded centered S@9 in this pilot; magnitude is a plausible contributor, not a proved sole cause.' if n>=center and n>raw else 'Matching centered-update magnitude did not recover a clear centered-policy advantage; a magnitude-only explanation is not established.')
  if 'action_gate_first' in methods:
   g=methods['action_gate_first']['counts']['s9'];bs=[methods[k]['counts']['s9'] for k in ['raw_first_only','shuffled_gate_first','uniform_gate_first']]
   statements.append('Action-aware gating beat all first-only controls in this pilot; prioritize independent replication before training a relevance gate.' if g>max(bs) else 'Action-distance gating did not beat every amplitude/shuffle/ungated first-only control; do not promote a locality architecture based on this proxy.')
  else:statements.append('Action-distance association failed the predeclared directional-repeat screen; gate evaluation skipped. This does not rule out other measures of relevance.')
  out['panels'][panel]={'methods':methods,'paired':comparisons,'diagnostics':d['panels'][panel],'interpretation':statements};lines.extend([f'## {panel}','','| Policy | S@1 | S@3 | S@5 | S@9 |','|---|---:|---:|---:|---:|'])
  for k,v in methods.items():lines.append('| '+k+' | '+' | '.join(str(v['counts'][f's{j}']) for j in [1,3,5,9])+' |')
  lines+=['']+statements+['']
 lines+=['## Limits','']+plan['limits'];write(R/'FINAL_SUMMARY.json',out);(R/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({p:v['interpretation'] for p,v in out['panels'].items()},indent=2))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['decision','final']);a=a.parse_args();decision() if a.mode=='decision' else final()
