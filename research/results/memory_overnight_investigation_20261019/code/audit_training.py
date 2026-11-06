"""Audit current Q0-first training episodes only; no legacy small pools."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
P=Path('/path/to/phyre-unc');S=P/'results/q0_first_feedback_20261018';D=P/'results/selector_feedback_pilot_20261016';R=P/'results/memory_overnight_investigation_20261019'
read=lambda p:json.loads(p.read_text());plan=read(S/'PLAN.json');out={'source':'q0_first_feedback_20261018 only','source_plan_sha256':hashlib.sha256((S/'PLAN.json').read_bytes()).hexdigest(),'panels':{}}
for panel,spec in plan['panels'].items():
 pools={}
 for task in spec['train_tasks']+spec['tasks']:
  with np.load(D/'pools'/panel/task.replace(':','_')/'pool.npz') as z:pools[task]={'labels':dict(zip(map(int,z['action_idx']),map(int,z['status']))),'fraction':float((z['status']>0).mean()),'n':len(z['status'])}
 arms={}
 for kind in ['residual','zero']:
  journal=S/'models'/panel/kind/'episodes.jsonl';slots=[json.loads(line) for line in journal.read_text().splitlines()];events=[e for row in slots for e in row['events']];active=[e for e in events if e['used_updates']>0]
  qcounts=[];poolfrac=[]
  for e in active:
   labels=pools[e['task']]['labels'];assert len(labels)==e['candidate_count']==1024
   assert set(e['supports'])<=set(e['selected']) and not set(e['query'])&set(e['selected'])
   assert all(labels[j]<0 for j in e['supports'])
   qcounts.append(sum(labels[j]>0 for j in e['query']));poolfrac.append(e['candidate_positive_fraction'])
  def desc(values):return {'n':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),'min':float(np.min(values)),'max':float(np.max(values))}
  arms[kind]={'slots':len(slots),'episodes':len(events),'active_episodes':len(active),'used_update_hist_all':dict(Counter(e['used_updates'] for e in events)),
    'used_update_hist_active':dict(Counter(e['used_updates'] for e in active)),'query_positive_count_hist':dict(Counter(qcounts)),
    'query_positive_fraction':sum(qcounts)/(8*len(active)),'active_episodes_without_positive_query':sum(n==0 for n in qcounts),
    'active_pool_positive_fraction':desc(poolfrac),'train_pool_positive_fraction_scene_macro':desc([pools[t]['fraction'] for t in spec['train_tasks']]),
    'eval_pool_positive_fraction_scene_macro':desc([pools[t]['fraction'] for t in spec['tasks']]),'journal_sha256':hashlib.sha256(journal.read_bytes()).hexdigest()}
 out['panels'][panel]=arms
(R/'TRAINING_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
