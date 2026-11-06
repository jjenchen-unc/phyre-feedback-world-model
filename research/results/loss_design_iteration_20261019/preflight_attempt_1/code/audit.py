from common import *
import argparse,numpy as np

def run(panel,kind,seed):
 plan=checked();folder=R/'models'/f'seed{seed}'/panel/kind;ck=read(folder/'complete.json');assert ck['slots']==128 and ck['plan_sha256']==sha(R/'PLAN.json') and sha(ck['checkpoint'])==ck['checkpoint_sha256']
 rows=[json.loads(l) for l in (folder/'episodes.jsonl').read_text().splitlines()];assert [x['slot'] for x in rows]==list(range(128));assert ck['optimizer_steps']==sum(r['optimizer_step'] for r in rows)
 pools={};active=0
 for i,row in enumerate(rows):
  assert row['optimizer_step']==any(e['supports'] for e in row['events'])
  for offset,e in enumerate(row['events']):
   if e['task'] not in pools:
    with np.load(P/'results/selector_feedback_pilot_20261016/pools'/panel/e['task'].replace(':','_')/'pool.npz') as z:pools[e['task']]={k:z[k].copy() for k in ['action_idx','status']}
   pool=pools[e['task']];labels=dict(zip(map(int,pool['action_idx']),map(int,pool['status'])))
   rng=np.random.default_rng([20261008,0,100000*seed+2*i+offset,0]);tasks=plan['panels'][panel]['train_tasks'];task=tasks[int(rng.integers(len(tasks)))];cap=int(rng.choice([1,2,4,8]));perm=rng.permutation(1024);assert task==e['task'] and cap==e['cap']
   q=[int(pool['action_idx'][j]) for j in perm if int(pool['action_idx'][j]) not in set(e['selected'])][:8];assert q==e['query'] and not set(q)&set(e['selected'])
   assert e['solved']==[labels[j]>0 for j in e['selected']] and bool(e['supports'])==(not e['solved'][0]);assert len(e['selected'])==len(set(e['selected'])) and e['supports']==e['selected'][:len(e['supports'])]
   assert all(labels[j]<0 for j in e['supports']) and e['selected'][0]==e['first_q0_argmax'];active+=bool(e['supports'])
 if seed==0:
  ref=read(B/'models'/panel/'uniform/complete.json')
  for key in ['initial_sha256','reader_sha256','optimizer_steps']:assert ck[key]==ref[key]
 else:
  paired=R/'models'/f'seed{seed}'/panel/'l1/complete.json'
  if paired.exists():
   ref=read(paired)
   for key in ['initial_sha256','reader_sha256','optimizer_steps']:assert ck[key]==ref[key]
 out={'complete':True,'panel':panel,'kind':kind,'seed':seed,'slots':128,'active_episodes':active,'optimizer_steps':ck['optimizer_steps'],'initial_sha256':ck['initial_sha256'],'reader_sha256':ck['reader_sha256'],'journal_sha256':sha(folder/'episodes.jsonl'),'checkpoint_sha256':ck['checkpoint_sha256']};write(R/'audits'/f'seed{seed}'/panel/f'{kind}.json',out);print(json.dumps(out))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--panel',required=True);a.add_argument('--kind',required=True);a.add_argument('--seed',type=int,default=0);a=a.parse_args();run(a.panel,a.kind,a.seed)
