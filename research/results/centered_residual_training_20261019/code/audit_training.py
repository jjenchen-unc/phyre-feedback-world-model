"""Validate centered training against the completed uniform128 control."""
import argparse,json
from pathlib import Path
import numpy as np
import train as tr
P,R,B=tr.P,tr.R,tr.B
read,write,sha=tr.read,tr.write,tr.sha

def run(panel):
 plan=read(R/'PLAN.json');folder=R/'models'/panel/'centered';ck=read(folder/'complete.json');refck=read(B/'models'/panel/'uniform/complete.json')
 assert ck['slots']==128 and ck['plan_sha256']==sha(R/'PLAN.json') and sha(ck['checkpoint'])==ck['checkpoint_sha256']
 for key in ['initial_sha256','reader_sha256','optimizer_steps']:assert ck[key]==refck[key],key
 rows=[json.loads(l) for l in (folder/'episodes.jsonl').read_text().splitlines()];refs=[json.loads(l) for l in (B/'models'/panel/'uniform/episodes.jsonl').read_text().splitlines()]
 assert [x['slot'] for x in rows]==list(range(128)) and sum(x['optimizer_step'] for x in rows)==ck['optimizer_steps']
 pools={};active=0;support_count=0
 for i,(row,ref) in enumerate(zip(rows,refs)):
  assert row['optimizer_step']==ref['optimizer_step'] and len(row['events'])==2
  for offset,(e,re) in enumerate(zip(row['events'],ref['events'])):
   assert (e['task'],e['cap'],e['selected'][0],bool(e['supports']))==(re['task'],re['cap'],re['selected'][0],bool(re['supports']))
   if e['task'] not in pools:
    with np.load(P/'results/selector_feedback_pilot_20261016/pools'/panel/e['task'].replace(':','_')/'pool.npz') as z:pools[e['task']]={k:z[k].copy() for k in ['action_idx','status']}
   pool=pools[e['task']];labels=dict(zip(map(int,pool['action_idx']),map(int,pool['status'])))
   rng=np.random.default_rng([20261008,0,2*i+offset,0]);tasks=plan['panels'][panel]['train_tasks'];task=tasks[int(rng.integers(len(tasks)))];cap=int(rng.choice([1,2,4,8]));perm=rng.permutation(1024)
   assert task==e['task'] and cap==e['cap']
   expected=[int(pool['action_idx'][j]) for j in perm if int(pool['action_idx'][j]) not in set(e['selected'])][:8]
   assert expected==e['query'] and len(set(e['query']))==8 and not set(e['query'])&set(e['selected'])
   assert e['solved']==[labels[j]>0 for j in e['selected']] and len(e['selected'])==len(set(e['selected']))
   assert e['used_updates']==len(e['supports']) and all(labels[j]<0 for j in e['supports']) and set(e['supports'])<=set(e['selected'])
   assert bool(e['supports'])== (not e['solved'][0])
   assert e['selected'][0]==e['first_q0_argmax'] and e['first_rank_source']=='q0_predictor'
   assert e['query_positive_count']==sum(labels[j]>0 for j in e['query'])
   active+=bool(e['supports']);support_count+=len(e['supports'])
 out={'complete':True,'panel':panel,'slots':128,'episodes':256,'active_episodes':active,'support_updates':support_count,'optimizer_steps':ck['optimizer_steps'],'reference_uniform128_draw_activity_steps_exact':True,'uniform_queries_verified':True,'initial_and_reader_exact':True,'training_plan_sha256':sha(R/'PLAN.json'),'checkpoint_sha256':ck['checkpoint_sha256'],'journal_sha256':sha(folder/'episodes.jsonl'),'elapsed_s':ck['elapsed_s']}
 write(R/f'TRAINING_AUDIT_{panel}.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--panel',required=True);run(p.parse_args().panel)
