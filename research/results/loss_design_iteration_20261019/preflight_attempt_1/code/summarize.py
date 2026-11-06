from common import *
from evaluate import collect
import argparse,numpy as np

def paired(a,b):
 assert set(a['first_success'])==set(b['first_success']);out={}
 for k in [1,3,5,9]:
  success=lambda d,t:d['first_success'][t] is not None and d['first_success'][t]<=k
  gains=[t for t in a['first_success'] if success(a,t) and not success(b,t)];losses=[t for t in a['first_success'] if success(b,t) and not success(a,t)]
  templates=sorted({t.split(':')[0] for t in a['first_success']})
  out[str(k)]={'gains':gains,'losses':losses,'net':len(gains)-len(losses),'leave_one_template_out_net':{x:sum(t.split(':')[0]!=x for t in gains)-sum(t.split(':')[0]!=x for t in losses) for x in templates}}
 return out

def round1():
 checked();out={'complete':True,'panels':{},'plan_sha256':sha(R/'PLAN.json')};decision={'complete':True,'panels':{},'rule':read(R/'PLAN.json')['round2_rule']}
 controls=read(C/'SUMMARY.json')['panels']
 for panel in ['cross','within']:
  methods={'l1':controls[panel]['methods']['raw_trained_raw'],**{k:collect(panel,k,0,'raw') for k in ['mse','smooth']}}
  winner=max(['l1','mse','smooth'],key=lambda k:tuple(methods[k]['counts'][s] for s in ['s9','s5','s3']))
  qualified=winner!='l1' and methods[winner]['counts']['s9']>methods['l1']['counts']['s9'] and methods[winner]['counts']['s5']>=methods['l1']['counts']['s5']
  decision['panels'][panel]={'winner':winner,'replicate':qualified,'counts':{k:v['counts'] for k,v in methods.items()}}
  out['panels'][panel]={'methods':methods,'paired':{k+'_vs_l1':paired(methods[k],methods['l1']) for k in ['mse','smooth']}}
 write(R/'ROUND1.json',out);write(R/'DECISION.json',decision);print(json.dumps(decision,indent=2))

def final():
 checked();r1=read(R/'ROUND1.json');decision=read(R/'DECISION.json');out={'complete':True,'plan_sha256':sha(R/'PLAN.json'),'panels':{},'decision':decision,'limits':read(R/'PLAN.json')['selection_limits']};text=['# Loss design iterations — results','', 'Same frozen Q0, classifier top-1024 candidate pool, Q0 first attempt, 9-attempt budget. All new updaters use raw residual during training. Each protocol has 100 evaluation scenes.','']
 for panel in ['cross','within']:
  methods=dict(r1['panels'][panel]['methods']);methods.update({'l1_centered':read(C/'SUMMARY.json')['panels'][panel]['methods']['raw_trained_centered'],'l1_zero':read(C/'SUMMARY.json')['panels'][panel]['methods']['raw_trained_zero']})
  for kind in ['mse','smooth']:
   for mode in ['centered','zero']:methods[kind+'_'+mode]=collect(panel,kind,0,mode)
  rep={}
  if decision['panels'][panel]['replicate']:
   winner=decision['panels'][panel]['winner'];rep={k:collect(panel,k,1,'raw') for k in ['l1',winner]}
   for key in ['initial_sha256','reader_sha256','optimizer_steps']:assert read(R/'models/seed1'/panel/'l1/complete.json')[key]==read(R/'models/seed1'/panel/winner/'complete.json')[key]
  diagnostics=read(R/f'DIAGNOSTIC_{panel}.json');diag={}
  for kind in ['mse','smooth']:
   records=read(R/f'DIAGNOSTIC_{panel}_{kind}.json');assert records['complete'];diag[kind]={}
   for split in ['train','evaluation']:
    rows=[x for x in records['rows'] if x['split']==split];refs=[next(r for r in diagnostics['rows'] if r['task']==x['task'] and r['split']==split)['methods']['raw'] for x in rows]
    diag[kind][split]={'n':len(rows),'mean_changes_vs_l1':{m:float(np.mean([x['metrics'][m]-r[m] for x,r in zip(rows,refs)])) for m in ['l1','mse','geometry','mean_bias_mse','centered_mse','update_norm']},'next_successes':sum(x['metrics']['next_success'] for x in rows),'l1_next_successes':sum(r['next_success'] for r in refs),'top8_successes':sum(x['metrics']['top8_success'] for x in rows),'l1_top8_successes':sum(r['top8_success'] for r in refs)}
  comparisons={k+'_vs_l1':paired(methods[k],methods['l1']) for k in ['mse','smooth']}
  for k in ['mse','smooth']:comparisons[k+'_raw_vs_zero']=paired(methods[k],methods[k+'_zero']);comparisons[k+'_centered_vs_raw']=paired(methods[k+'_centered'],methods[k])
  repcompare=paired(rep[decision['panels'][panel]['winner']],rep['l1']) if rep else None
  if rep:
   d=repcompare['9']['net'];verdict='Candidate improved S@9 in both updater seeds; prioritize a new fold before claiming robustness.' if d>0 else 'Round1 improvement did not replicate as a positive S@9 difference; no robust winner established.'
  else:verdict='No new raw-input loss met the predeclared S@9 gain without S@5 regression; no second training round launched. Use fixed-query and input controls to guide next work.'
  out['panels'][panel]={'methods':methods,'paired':comparisons,'seed1':rep,'seed1_paired':repcompare,'diagnostics':diag,'interpretation':verdict}
  text.extend([f'## {panel}','','| Method, seed0 | S@1 | S@3 | S@5 | S@9 |','|---|---:|---:|---:|---:|'])
  for k,v in methods.items():text.append('| '+k+' | '+' | '.join(str(v['counts'][f's{n}']) for n in [1,3,5,9])+' |')
  text.extend(['',verdict,''])
  if rep:
   text.extend(['| Method, seed1 | S@1 | S@3 | S@5 | S@9 |','|---|---:|---:|---:|---:|'])
   for k,v in rep.items():text.append('| '+k+' | '+' | '.join(str(v['counts'][f's{n}']) for n in [1,3,5,9])+' |')
  text.extend(['','Fixed-query diagnostics compare identical first support, observation and eight query actions. Negative error changes mean lower error, but do not establish better closed-loop success.','', '```json',json.dumps(diag,indent=2),'```',''])
 text.extend(['## Interpretation limits','','Loss coefficients were fixed using training-only final-memory-state gradient norms. This does not guarantee equal full-parameter gradients throughout training. Seed1 changes updater initialization and episode sampling, not the pretrained Q0 or fold. Centered/zero evaluation inputs are interventions. Zero-input performance is not proof of semantic feedback use or non-use. No training-loss-only winner is selected.'])
 write(R/'FINAL_SUMMARY.json',out);(R/'REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps({p:d['interpretation'] for p,d in out['panels'].items()},indent=2))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['round1','final']);a=a.parse_args();round1() if a.stage=='round1' else final()
