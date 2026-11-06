from common import *
p=read(S/'PLAN.json')
train_cases={}
for panel in ['cross','within']:
 rows=[json.loads(x) for x in (B/'models'/panel/'uniform/episodes.jsonl').read_text().splitlines()];cases=[];seen=set()
 for row in rows:
  for offset,e in enumerate(row['events']):
   if e['supports'] and e['task'] not in seen:
    seen.add(e['task']);cases.append({'index':2*row['slot']+offset,'task':e['task']})
 train_cases[panel]=cases[:8];assert len(train_cases[panel])==8
plan={'panels':p['panels'],'slots':128,'episodes_per_slot':2,'query_count':8,'budget':9,'training_calibration_cases':train_cases,
 'round1':{'new_losses':['mse','smooth'],'baseline':'Existing raw uniform128 L1 seed7919; exact loss+gradient and initialization replay required','train_input':'raw residual','frozen':'Same Q0/initialtokens/probe/scorer; no base retraining','loss':'scale*latent_loss + .1geometry; uniform8untried, fullBPTT, same AdamW/OneCycle128'},
 'calibration':'Training-only8distinctactive scenes, fixed first failedsupport/currentmemory and8uniformqueries. SmoothL1 beta=median absolute query error from frozen rawtrained128 reference. Per-loss scale=median ratio of L1 gradient norm to candidate gradient norm with respect to final memory state, measured on same reference cases; frozen before fresh training. This matches state-gradient scale approximately, not full training gradients. Log parameter-gradient check separately.',
 'diagnostic':'Beforetraining compare Q0/rawL1/rawL1centered/newcentertrained on fixed samefirstsupport and8queries, train8/eval8 scenes protocol. Record L1,MSE,featuremeanbias,centeredMSE,geometry,top-tailenergy,full1024ranking andupdate norm. No result-dependent taskselection.',
 'round2_rule':'After round1 raw-input100scene evaluation, choose perprotocol loss maximizing (S9,S5,S3), preferring L1 on exactties. A newloss qualifies for seed1 replication only if S9>L1 and S5>=L1. If anyqualify, train paired freshL1 and qualifyingloss for eachqualifyingprotocol with initializer7920 and independent deterministic episode draws;same frozenQ0. If none, no extra training. In allcases evaluate round1 MSE/SmoothL1 with centered andzero inputs to test whether improvement needs residualcontent.',
 'max_new_trainings':8,'selection_limits':['Singlefold andsameQ0;seed1replicatesupdaterinitialization/samplingonly','Adaptive diagnostics do not prove causality','Training histories change with learnedpolicy','Loss magnitude andgradient scale not assumed equivalent','No semantic claim from zero-intervention alone'],
 'dependencies':{str(f):sha(f) for f in [S/'PLAN.json',B/'PLAN.json',C/'PLAN.json',C/'code/train.py',*sorted((R/'code').glob('*.py'))]}}
for panel in ['cross','within']:
 tasks=[]
 for task in sorted(p['panels'][panel]['tasks'],key=lambda t:hashlib.sha256(('loss-fixed-panel-v1'+t).encode()).hexdigest()):
  row=read(S/'evaluation'/panel/'q0'/task.replace(':','_')/'result.json')
  if not row['events'][0]['solved']:tasks.append(task)
 plan.setdefault('diagnostic_eval_tasks',{})[panel]=tasks[:8]
if (R/'PLAN.json').exists():assert read(R/'PLAN.json')==plan
else:write(R/'PLAN.json',plan)
print('REGISTERED',sha(R/'PLAN.json'))
