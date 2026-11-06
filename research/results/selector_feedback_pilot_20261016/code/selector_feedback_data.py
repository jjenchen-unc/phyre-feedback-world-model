"""Prepare outcome-independent full100k selector pools for the feedback pilot."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

P=Path('/path/to/phyre-unc')
R=P/'results/selector_feedback_pilot_20261016'
C=P/'results/candidate_coverage_100k_20261016'
sys.path.insert(0,str(R/'code'))
import deploy_pool_data as data
data.R=R
import candidate_coverage_100k as coverage
coverage.R=R
read,write,sha,save=data.read,data.write,data.sha,data.save_npz


def choose_trial(scores,tried,turn):
    """The first selector choice is fixed; later choices use current scores only."""
    assert int(np.count_nonzero(tried))==turn
    if turn==0:
        assert not tried[0]
        return 0
    assert scores is not None and len(scores)==len(tried) and not tried.all()
    values=np.array(scores,copy=True);values[tried]=-np.inf
    return int(values.argmax())


def prepare():
    import phyre
    if (R/'PLAN.json').exists():
        data.checked();return
    summary=read(C/'SUMMARY.json');source=read(C/'PLAN.json')
    assert summary['complete'] and summary['plan_sha256']==sha(C/'PLAN.json')
    parent=read(data.OLD/'PLAN.json')
    plan=copy.deepcopy(parent)
    plan.update(scope='Short pilot: full100k winning selector top1 first trial, feedback top1024 reranking',
                date='2026-10-16',candidate_actions=100000,top_k=1024,
                train_scenes_per_panel=64,slots=256,episodes_per_slot=2,budget=9,
                methods=['q0','residual','zero','tta'],seed=0,
                dependencies=copy.deepcopy(parent['dependencies']),coverage_source=str(C/'SUMMARY.json'),
                labels='Pool outcomes re-simulated AFTER selector scores/top1024 fixed; never label-balanced',
                simulation_precision='Preserve official float64 actions for rendering/simulation; model action inputs float32',
                first_attempt='Selector argmax, identical across all four methods; not Q0 argmax',
                rerank_tie_break='Stable selector pool order among equal current scores',
                zero='Zero residual plus selected action/current state; no full observation required for support',
                q0='No weight/token update; one cached reranking after forced selector first attempt',
                extra_control='Selector static original order, computed from existing pool outcomes without additional training',
                training_loss='Final updated-state future latent L1 + 0.1 geometry; full BPTT',
                checkpoint_choice='User-specified corresponding lambda0 cross/lambda0.03 within seed0; not optimality claim')
    plan.pop('gpu_uuids',None)
    plan.pop('gpu_limit',None)
    plan['resources']={'pool_collection_gpus':40,'training_and_evaluation_gpus':6}
    plan['evaluation']={'tasks_per_panel':100,'max_attempts':9,'source':'Winning selector complete100k top1024',
                        'first_attempt':'Forced selector top1 for all four arms','reuse_old_evaluation_metrics':False}
    plan['training']={'slots':256,'episodes_per_slot':2,'scenes_per_panel':64,
                      'support_caps':[1,2,4,8],'queries':8,'query_times':8,
                      'first_attempt':'selector top1','updater_geometry_weight':.1,
                      'optimizer':'AdamW lr3e-4 wd.05 clip1, OneCycle3pct over256slots'}
    cache=phyre.get_default_100k_cache('ball')
    np.testing.assert_array_equal(cache.action_array,np.load(C/'inputs/actions.npz')['actions'])
    save(R/'inputs/actions.npz',actions=cache.action_array)
    for panel,spec in plan['panels'].items():
        result=summary['panels'][panel]
        counts={k:v['covered_tasks'] for k,v in result['methods'].items()}
        # Full classifier score time vs full base time: only used if coverage ties.
        means={m:float(np.mean([read(f)['seconds_this_invocation'] for f in (C/'scores'/panel/m).glob('*/complete.json')])) for m in ['base','scene']}
        selector=sorted(counts,key=lambda m:(-counts[m],means[m],m))[0]
        spec.update(source['panels'][panel]);spec['selector']=selector
        spec.pop('cases',None);spec.pop('source_counts',None)
        spec.pop('store_files_sha256',None)
        for initial in spec['initial'].values():initial.pop('evaluation',None)
        spec['selector_decision']={'covered_tasks':counts,'mean_rank_s':means,'rule':'higher coverage, then lower measured ranking time'}
        for task in spec['tasks']:
            for method in set(['base',selector]):
                folder=C/'scores'/panel/method/task.replace(':','_')
                receipt=read(folder/'complete.json');path=folder/'scores.npz'
                assert sha(path)==receipt['scores_sha256'];plan['dependencies'][str(path)]=receipt['scores_sha256']
        token_file=C/'inputs'/(panel+'_tokens.npz')
        assert sha(token_file)==source['dependencies'][str(token_file)]
        plan['dependencies'][str(token_file)]=sha(token_file)
        # Spread the small pilot over templates; no outcome information enters.
        manifest=read(spec['source_manifest'])
        tasks=sorted(row['task'] for row in manifest['rows'] if row['split']=='train')
        by={t[:5]:[] for t in tasks}
        for t in tasks:by[t[:5]].append(t)
        rng=np.random.default_rng(20261016)
        for v in by.values():rng.shuffle(v)
        selected=[]
        for depth in range(max(map(len,by.values()))):
            for key in sorted(by):
                if depth<len(by[key]):selected.append(by[key][depth])
        spec['train_tasks']=selected[:64]
        assert len(set(spec['train_tasks']))==64 and not set(spec['train_tasks'])&set(spec['tasks'])
        if panel=='cross':assert not {t[:5] for t in spec['train_tasks']}&{t[:5] for t in spec['tasks']}
        sim=phyre.initialize_simulator(spec['train_tasks'],'ball')
        save(R/'inputs'/(panel+'_train_scenes.npz'),tasks=np.array(spec['train_tasks']),frames=np.asarray(sim.initial_scenes))
        statuses=np.stack([cache.load_simulation_states(t) for t in spec['train_tasks']])
        save(R/'inputs'/(panel+'_train_labels.npz'),tasks=np.array(spec['train_tasks']),statuses=statuses)
        for arch in ['base','q0']:
            ck=Path(spec['initial'][arch]['checkpoint']);assert sha(ck)==spec['initial'][arch]['sha256']
            plan['dependencies'][str(ck)]=sha(ck)
        cp=Path(spec['classifier_checkpoint']);plan['dependencies'][str(cp)]=sha(cp)
    for f in [C/'PLAN.json',C/'SUMMARY.json',data.OLD/'PLAN.json']+list((R/'code').glob('*.py'))+list((R/'inputs').glob('*.npz')):
        plan['dependencies'][str(f)]=sha(f)
    write(R/'PLAN.json',plan)
    print('PREPARED',{p:s['selector'] for p,s in plan['panels'].items()},flush=True)


def encode_scenes():
    import ast
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    torch.set_num_threads(2)
    assert 'A6000' in torch.cuda.get_device_name(0)
    plan=data.checked();dev=torch.device('cuda')
    tree=ast.parse((R/'code/scene_classifier.py').read_text())
    node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='encode']
    env={'torch':torch,'F':F,'np':np,'dev':dev,
         'WAD':torch.tensor([[255,255,255],[243,79,70],[107,206,187],[24,119,242],[75,74,164],[185,202,210],[0,0,0],[252,223,227]],device=dev)/255,
         'MEAN':torch.tensor([.485,.456,.406],device=dev).view(1,3,1,1),
         'STD':torch.tensor([.229,.224,.225],device=dev).view(1,3,1,1)}
    exec(compile(ast.Module(body=node,type_ignores=[]),'<encode>', 'exec'),env)
    enc=AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large',trust_remote_code=True,
          revision='e831a0347737fcaa660b39c57d41c109de399845',local_files_only=True).eval().cuda().half()
    for panel,spec in plan['panels'].items():
        with np.load(R/'inputs'/(panel+'_train_scenes.npz')) as z:
            assert z['tasks'].tolist()==spec['train_tasks'];start=time.time();tokens=env['encode'](z['frames'],enc)
            save(R/'inputs'/(panel+'_train_tokens.npz'),tokens=tokens.numpy(),tasks=z['tasks'])
        write(R/'inputs'/(panel+'_encoding.json'),{'plan':sha(R/'PLAN.json'),'seconds':time.time()-start,
              'token_sha256':sha(R/'inputs'/(panel+'_train_tokens.npz'))})
    print('SCENES_ENCODED',flush=True)


def build(panel,task,split,plan,encoder,renderer,label_worker,models):
    folder=data.folder_for(panel,task)
    if (folder/'complete.json').exists():data.load_pool(panel,task);return
    folder.mkdir(parents=True,exist_ok=True)
    spec=plan['panels'][panel];selector=spec['selector']
    actions=np.load(R/'inputs/actions.npz')['actions'];start=time.time()
    if split=='eval':
        with np.load(C/'scores'/panel/selector/task.replace(':','_')/'scores.npz') as z:scores=z['scores'].copy()
        with np.load(C/'scores'/panel/'base'/task.replace(':','_')/'scores.npz') as z:valid=z['rendered_valid']
        ids=np.flatnonzero(valid)
    else:
        i=spec['train_tasks'].index(task)
        labels=np.load(R/'inputs'/(panel+'_train_labels.npz'))['statuses'][i]
        if selector=='scene':
            with np.load(R/'inputs'/(panel+'_train_tokens.npz')) as z:token=z['tokens'][i]
            scores=coverage.scene_scores(models['scene'],token,actions)
            ids=np.flatnonzero(labels!=0)
        else:
            partial=folder/'selection_partial.npz'
            if partial.exists():
                with np.load(partial) as z:scores=z['scores'];valid=z['valid'];begin=int(z['next_index'])
            else:scores=np.full(100000,-np.inf);valid=np.zeros(100000,bool);begin=0
            for b in range(begin,100000,512):
                raw=renderer.render(task,np.arange(b,min(b+512,100000)))
                sc=coverage.base_scores(models['base'],models['scorer'],encoder,raw,actions)
                scores[raw['indices']]=sc;valid[raw['indices']]=True
                save(partial,scores=scores,valid=valid,next_index=min(b+512,100000))
            ids=np.flatnonzero(valid)
    assert np.isfinite(scores[ids]).all()
    order=ids[np.argsort(-scores[ids],kind='stable')]
    # Exact validity check of classifier proposals; fill from next-ranked legal actions.
    selection_s=time.time()-start;render_start=time.time()
    raws=[];count=cursor=0;rejected=set()
    while count<1024 and cursor<len(order):
        proposed=order[cursor:cursor+min(256,1024-count)]
        cursor+=len(proposed)
        raw=renderer.render(task,proposed)
        rejected.update(set(map(int,proposed))-set(map(int,raw['indices'])))
        if len(raw['indices']):raws.append(raw)
        count+=len(raw['indices'])
    assert count==1024,(task,count)
    raw={k:np.concatenate([x[k] for x in raws]) for k in ['indices','frames','features']}
    selected=raw['indices'];render_s=time.time()-render_start
    encode_start=time.time();z0=encoder.initial(raw['frames']);encode_s=time.time()-encode_start
    label_start=time.time()
    label_path=folder/'labels.tmp.npy'
    label_worker.request(mode='labels',task=task,actions=actions[selected].tolist(),out=str(label_path))
    status=np.load(label_path);assert (status!=0).all()
    data.save_npy(folder/'z0.npy',z0)
    save(folder/'pool.npz',action_idx=selected,action=actions[selected].astype(np.float32),simulation_action=actions[selected],
         features=raw['features'],selection_score=scores[selected],status=status)
    save(folder/'selector_scores.npz',scores=scores,
         ranked_ids=np.array([j for j in order if int(j) not in rejected]),
         rejected=np.array(sorted(rejected),np.int64))
    write(folder/'complete.json',{'plan':sha(R/'PLAN.json'),'task':task,'split':split,'selector':selector,
          'candidates':1024,'positive':int((status>0).sum()),'first_action_idx':int(selected[0]),
          'seconds':time.time()-start,'selection_s':selection_s,'retained_initial_render_s':render_s,
          'retained_initial_encode_s':encode_s,'offline_label_and_write_s':time.time()-label_start,
          'pool_sha256':sha(folder/'pool.npz'),'initial_sha256':sha(folder/'z0.npy')})
    label_path.unlink()
    if (folder/'selection_partial.npz').exists():(folder/'selection_partial.npz').unlink()
    print('POOL',panel,split,task,selector,time.time()-start,flush=True)


def collect(shard,nshards):
    import torch
    torch.set_num_threads(2)
    plan=data.checked();encoder=data.EncoderProxy();renderer=coverage.Renderer();worker=data.CPUWorker()
    try:
        jobs=[(p,t,s) for p,spec in plan['panels'].items() for s,key in [('train','train_tasks'),('eval','tasks')] for t in spec[key]][shard::nshards]
        for panel in plan['panels']:
            models={}
            if plan['panels'][panel]['selector']=='base':
                models['base'],models['scorer'],unused=coverage.setup_base(panel,plan);unused.close()
            else:
                models['scene']=coverage.scene_model(panel,plan)
                encoding=read(R/'inputs'/(panel+'_encoding.json'))
                assert encoding['plan']==sha(R/'PLAN.json')
                assert encoding['token_sha256']==sha(R/'inputs'/(panel+'_train_tokens.npz'))
                # Same classifier arithmetic on A5000 collection and A6000 coverage.
                token=np.load(C/'inputs'/(panel+'_tokens.npz'))['tokens'][0]
                actions=np.load(R/'inputs/actions.npz')['actions'][:1024]
                actual=coverage.scene_scores(models['scene'],token,actions)
                with np.load(C/'scores'/panel/'scene'/plan['panels'][panel]['tasks'][0].replace(':','_')/'scores.npz') as z:
                    np.testing.assert_array_equal(actual,z['scores'][:1024])
            for p,task,split in jobs:
                if p==panel:build(p,task,split,plan,encoder,renderer,worker,models)
            del models;torch.cuda.empty_cache()
    finally:encoder.close();renderer.close();worker.close()
    write(R/'collection'/f'{shard}.json',{'complete':True,'plan':sha(R/'PLAN.json')})


def audit():
    plan=data.checked();report={}
    for panel,spec in plan['panels'].items():
        rows=[]
        for split,key in [('train','train_tasks'),('eval','tasks')]:
            for task in spec[key]:
                pool=data.load_pool(panel,task);folder=data.folder_for(panel,task);receipt=read(folder/'complete.json')
                assert sha(folder/'pool.npz')==receipt['pool_sha256'] and sha(folder/'z0.npy')==receipt['initial_sha256']
                assert len(set(pool['action_idx']))==1024 and (pool['action_idx']<100000).all()
                assert (pool['selection_score'][:-1]>=pool['selection_score'][1:]).all()
                with np.load(folder/'selector_scores.npz') as z:
                    np.testing.assert_array_equal(pool['selection_score'],z['scores'][pool['action_idx']])
                    assert list(z['ranked_ids']).index(pool['action_idx'][0])==0
                if split=='eval':
                    summary=read(C/'SUMMARY.json')['panels'][panel]
                    ref=next(v for v in summary['tasks'] if v['task']==task)[spec['selector']]['top1024_ids']
                    np.testing.assert_array_equal(pool['action_idx'],ref)
                rows.append({'task':task,'split':split,'positive':int((pool['status']>0).sum())})
        report[panel]=rows
    write(R/'DATA_AUDIT.json',{'passed':True,'plan':sha(R/'PLAN.json'),'panels':report})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','encode','collect','audit'])
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=40);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='encode':encode_scenes()
    elif a.mode=='collect':collect(a.shard,a.nshards)
    else:audit()
