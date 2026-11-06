"""Freeze common prefix/policy implementation and audit mixed support candidates."""
import json,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'query_token_dynamic_20261005'
sys.path.insert(0,str(P/'code'))
import fresh_memory_geometry as f
from redesign_lib import Store
assert not (R/'PREPARATION.json').exists()
m=json.loads((P/'manifest.json').read_text());inv=json.loads((R.parent/'support_utility_20261004/inventory.json').read_text())
store=Store(include_features=False)
assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest()==m['store_files_sha256']
by={r['task']:r for r in inv['rows']};rows=[]
for r in m['rows']:
    item=by[r['task']];support=item['support'];query=r['query']
    assert not set(support)&set(query)
    assert not {item['groups'][str(i)]['group'] for i in support}&{item['groups'][str(i)]['group'] for i in query}
    assert all(str(store.tasks[i])==r['task'] for i in support+query)
    rows.append(dict(task=r['task'],split=r['split'],candidates=support,query=query,times=r['times']))
train=[r for r in rows if r['split']=='train'];assert len(train)==1200
counts={'train_scenes':len(train),'train_candidate_actions':sum(len(r['candidates']) for r in train),
        'train_candidate_successes':sum(int(store.labels[r['candidates']].sum()) for r in train),
        'train_scenes_with_success_candidate':sum(bool(store.labels[r['candidates']].any()) for r in train),
        'train_scenes_with_failure_candidate':sum(bool((~store.labels[r['candidates']]).any()) for r in train),
        'train_queries':sum(len(r['query']) for r in train)}
output=subprocess.check_output([sys.executable,str(R/'code/test_prefix_policy_predictor.py')],env={**__import__('os').environ,'PYTHONPATH':str(P/'code')},text=True)
cpu=json.loads(output);assert cpu['passed']
f.write_json(R/'CPU_AUDIT.json',cpu)
f.write_json(R/'PREPARATION.json',{'scope':'Common implementation and data audit only; final experiment suite not yet selected',
    'parent_manifest_sha256':f.digest(P/'manifest.json'),'code':{p.name:f.digest(p) for p in (R/'code').glob('*.py')},
    'data_counts':counts,'rows':rows,'architecture':{'prefix':'32 slots concatenated on spatial axis, shared original spatial/temporal self-attention',
    'width':384,'no_read_adapters':True,'Q0_active_from_first_action':True,'candidate_forward_does_not_mutate_state':True},
    'policy':{'candidate_pool':'Existing support-half pool including successes and failures, query-group disjoint',
    'selection':'Current predictor/tokens plus fixed predicted_mix_s0 scorer at every trial; stable argmax, no repeats, stop on success',
    'residual':'Observed latent minus selected action current-state prediction; recomputed during full-BPTT replay',
    'no_policy_gradient':'Discrete action selection is detached; no gradient through argmax',
    'truth_cache_allowed':True,'residual_cache_allowed':False},
    'limits':['Not the full1024 deployment candidate pool during training; existing support half is smaller',
    'Positive-prefix updater supervision requires an observed failure; first-action successes must not be relabeled or bypassed',
    'Full-Q0 pretraining budget and ABCDE-vs-staged matrix must be fixed before formal training',
    'Common GPU smoke scheduled only after full ABCDE evaluation and its audit finish']})
print(json.dumps(counts,indent=2))
