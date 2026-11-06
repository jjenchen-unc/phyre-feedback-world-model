"""Execute base/random policies; pair actual failed prefixes and common queries.

Only selected simulator outcomes stop a policy. The cache's success labels
never prefilter attempts. Auxiliary query outcomes are targets, not history.
"""
import argparse
import json
from pathlib import Path
from multiprocessing import Pool
import time
import numpy as np
import phyre
from phyre import simulator as S


def work(item):
    task,root,scores_dir,data_dir=item;root=Path(root);data_dir=Path(data_dir)
    target=root/'traces'/f'{task.replace(":","_")}.json'
    if target.exists():return json.loads(target.read_text())
    c=np.load(Path(scores_dir)/f'{task.replace(":","_")}.npz');actions=c['action'];aids=c['action_idx']
    sim=phyre.initialize_simulator([task],'ball');seen={}
    def outcome(j):
        j=int(j)
        if j not in seen:
            seen[j]=int(sim.simulate_action(0,actions[j],need_images=False).status)
            assert seen[j]!=0,('invalid cached candidate',task,j)
        return seen[j]
    rng=np.random.default_rng([20260929,*map(int,task.split(':'))])
    def execute(order):
        tried=[];failures=[];terminal=None
        for j in order[:10]:
            j=int(j);status=outcome(j);tried.append({'local':j,'action_idx':int(aids[j]),'status':status,'score':float(c['scores'][j])})
            if status>0:terminal=j;break
            failures.append(j)
        return {'attempts':tried,'failures':failures,'terminal_success':terminal}
    base=execute(np.argsort(-c['scores'],kind='stable'));random=execute(rng.permutation(len(aids)))
    excluded=set(base['failures']+random['failures'])
    available=[int(j) for j in np.argsort(-c['scores'],kind='stable') if int(j) not in excluded]
    query=[];roles=[]
    # Retain terminal successes as explicitly labelled auxiliary supervision.
    for tr in [base,random]:
        j=tr['terminal_success']
        if j is not None and j not in query:query.append(j);roles.append('terminal_success_target')
    for j in available[:8]:
        if j not in query:query.append(j);roles.append('high_score_target')
    for j in rng.permutation(available):
        if len(query)>=32:break
        if int(j) not in query:query.append(int(j));roles.append('uniform_target')
    needed=sorted(set(query+base['failures']+random['failures']))
    manifest=[];raw=data_dir/'rollouts';raw.mkdir(parents=True,exist_ok=True)
    for j in needed:
        ai=int(aids[j]);status=outcome(j);f=raw/f'{task.replace(":","_")}_{ai:06d}.npz'
        if not f.exists():
            ui,ok=sim._action_mapper.action_to_user_input(actions[j]);assert ok
            sc=S.add_user_input_to_scene(sim._tasks[0].scene,ui,sim._keep_spaces)
            frames=S.simulate_scene(sc,1017)[::8]
            images=np.stack([S.scene_to_raster(x) for x in frames]).astype(np.uint8)
            features=np.concatenate([S.scene_to_featurized_objects(x).features for x in frames]).astype(np.float32)
            assert len(images)==128
            tmp=f.with_suffix('.tmp.npz');np.savez_compressed(tmp,frames=images,features=features,
                action=actions[j],action_idx=ai,status=status,cache_status=int(c['cache_status'][j]),steps=1017,stride=8,task=task);tmp.replace(f)
        else:
            with np.load(f) as z:
                assert int(z['status'])==status and len(z['frames'])==128
                assert np.allclose(z['action'],actions[j],atol=0,rtol=0)
        manifest.append({'task':task,'action_idx':ai,'status':status,'solved':status>0,
            'file':'rollouts/'+f.name,'local':j,'cache_status':int(c['cache_status'][j])})
    k=min(len(base['failures']),len(random['failures']))
    episodes=[{'attempt':j,'actual_k':min(j,4),'A':random['failures'][max(0,j-4):j],
               'B':base['failures'][max(0,j-4):j],'query':query} for j in range(1,k+1)]
    result={'task':task,'n_valid_candidates':int(c['n_valid_candidates']),'n_shortlist':len(aids),
        'base':base,'random':random,'query':query,'query_roles':roles,'episodes':episodes,
        'manifest':manifest,'query_successes':sum(outcome(j)>0 for j in query),
        'shortlist_success_fraction_offline':float((c['cache_status']>0).mean())}
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.json');tmp.write_text(json.dumps(result));tmp.replace(target)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--scores',required=True)
    p.add_argument('--data',default='/path/to/phyre-unc/data');p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();root=Path(a.root);selection=json.loads((root/'selection.json').read_text());t0=time.time()
    tasks=sorted(selection['train_tasks']+selection['validation_tasks']);rows=[]
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(work,[(t,a.root,a.scores,a.data) for t in tasks]):
            rows.append(r);print(json.dumps({'task':r['task'],'paired_episodes':len(r['episodes']),'seconds':round(time.time()-t0)}),flush=True)
    manifest=[m for r in sorted(rows,key=lambda x:x['task']) for m in r['manifest']]
    (Path(a.data)/'manifests/memory_deploy_pilot_20260929_rollouts.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in manifest))
    stats={'n_tasks':len(rows),'n_rollouts':len(manifest),'cache_disagreements':sum((m['status']>0)!=(m['cache_status']>0) for m in manifest),
        'train_eligible_scenes':sum(bool(r['episodes']) for r in rows if r['task'] in selection['train_tasks']),
        'validation_eligible_scenes':sum(bool(r['episodes']) for r in rows if r['task'] in selection['validation_tasks']),
        'paired_episodes':sum(len(r['episodes']) for r in rows),'seconds':round(time.time()-t0)}
    (root/'collection_summary.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
