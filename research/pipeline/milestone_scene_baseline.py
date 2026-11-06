"""Frozen historical initial-scene classifiers on the milestone's exact pools."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

P=Path('/path/to/phyre-unc')
R=P/'results/milestone_scene_baseline_20261015'
M=P/'results/milestone_memory_20261015'
D=Path('/mnt/meg/jjenchen/phyre-unc')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
def checked():
    plan=read(R/'PLAN.json')
    for path,digest in plan['dependencies'].items():assert sha(path)==digest,path
    return plan
def provenance():return {'plan':sha(R/'PLAN.json'),'code':{str(f):sha(f) for f in sorted((R/'code').glob('*.py'))}}


def prepare():
    assert not (R/'PLAN.json').exists()
    milestone=read(M/'PLAN.json');matched=read(P/'results/matched_base_q0_20261012/PLAN.json')
    dependencies={str(f):sha(f) for f in [M/'PLAN.json',M/'SUMMARY.json']};panels={}
    for panel in ['cross','within']:
        setup='ball_cross_template_f1' if panel=='cross' else 'ball_within_template_f0'
        stem='scene_clf_matched_s16r16_'+('cross_f1' if panel=='cross' else 'within')
        checkpoint=P/'results'/(stem+'.pt');record=P/'results'/(stem+'.json')
        manifest=P/'data/manifests'/(setup+'_train_s16r16.jsonl')
        rows=[json.loads(l) for l in manifest.read_text().splitlines()]
        meta=Path(matched['panels'][panel]['train_pack']+'_meta.npz')
        with np.load(meta) as z:
            assert len(rows)==len(z['files'])==matched['panels'][panel]['n_train']
            assert {r['file'] for r in rows}==set(z['files'].tolist())
            by={r['file']:r for r in rows}
            assert all(bool(y)==by[str(f)]['solved'] for f,y in zip(z['files'],z['solved']))
        prior=read(record)['aggregate'];assert prior['data_mode']=='matched'
        assert prior['n_train_pos']==sum(r['solved'] for r in rows)
        assert prior['n_train_neg']==sum(not r['solved'] for r in rows)
        tasks=milestone['panels'][panel]['tasks'];train_tasks={r['task'] for r in rows}
        assert len(tasks)==100 and not set(tasks)&train_tasks
        if panel=='cross':assert not {t[:5] for t in tasks}&{t[:5] for t in train_tasks}
        scene=D/'scenes'/(setup+'_dev.npz')
        with np.load(scene) as z:reference=str(z['tasks'][0])
        panels[panel]={'setup':setup,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
            'historical_record':str(record),'training_manifest':str(manifest),'training_pairs':len(rows),
            'historical_updates':prior['updates'],'tasks':tasks,'reference_scene_file':str(scene),
            'reference_task':reference,'cases':milestone['panels'][panel]['cases']}
        for f in [checkpoint,record,manifest,meta,scene]:dependencies[str(f)]=sha(f)
    write(R/'PLAN.json',{'panels':panels,'dependencies':dependencies,
        'methods':'Existing seed0 matched-s16r16 fixed classifier; no training or online update',
        'inputs':'Action-free initial scene 196x1024 LeVJEPA tokens plus candidate action',
        'encoder_revision':'e831a0347737fcaa660b39c57d41c109de399845',
        'batch':512,'budgets':[1,3,5,9],'pool':'Exact milestone base-shortlist 1024 per task',
        'limits':'Same base training pairs, different supervised target, representation and compute. Historical checkpoint stores only state_dict; provenance follows preserved records/manifests.'})
    print('PREPARED',flush=True)


def render():
    import phyre
    plan=checked();audit={}
    for panel,spec in plan['panels'].items():
        simulator=phyre.initialize_simulator([spec['reference_task']]+spec['tasks'],'ball')
        frames=np.asarray(simulator.initial_scenes)
        with np.load(spec['reference_scene_file']) as z:
            j=z['tasks'].tolist().index(spec['reference_task'])
            np.testing.assert_array_equal(frames[0],z['frames'][j])
            actions=z['actions']
        for task in spec['tasks']:
            case=spec['cases'][task];path=Path(case['shortlist'])
            assert sha(path)==case['shortlist_sha256']
            with np.load(path) as z:np.testing.assert_array_equal(z['action'],actions[z['action_idx']])
        path=R/'inputs'/(panel+'.npz')
        save_npz(path,tasks=np.array(spec['tasks']),frames=frames[1:])
        audit[panel]={'reference_initial_render_exact':True,'action_index_mapping_exact':True,
            'file':str(path),'sha256':sha(path),'frames':list(frames[1:].shape)}
    write(R/'INPUT_AUDIT.json',{'provenance':provenance(),'panels':audit});print('RENDER_PASS',flush=True)


def evaluate(panel):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel
    plan=checked();prov=provenance();spec=plan['panels'][panel];audit=read(R/'INPUT_AUDIT.json')
    assert audit['provenance']==prov
    inp=audit['panels'][panel];assert sha(inp['file'])==inp['sha256']
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);dev=torch.device('cuda')
    # Execute only the original class and encoder function, never its train main.
    source=ast.parse((R/'code/scene_classifier.py').read_text())
    nodes=[n for n in source.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['SceneClassifier','encode']]
    assert len(nodes)==2
    env={'torch':torch,'nn':nn,'F':F,'np':np,'dev':dev}
    env['WAD']=torch.tensor([[255,255,255],[243,79,70],[107,206,187],[24,119,242],
        [75,74,164],[185,202,210],[0,0,0],[252,223,227]],dtype=torch.float32,device=dev)/255
    env['MEAN']=torch.tensor([.485,.456,.406],device=dev).view(1,3,1,1)
    env['STD']=torch.tensor([.229,.224,.225],device=dev).view(1,3,1,1)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen_scene_classifier>','exec'),env)
    start=time.time()
    enc=AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large',trust_remote_code=True,
        revision=plan['encoder_revision'],local_files_only=True).eval().to(dev).half()
    with np.load(inp['file']) as z:
        assert z['tasks'].tolist()==spec['tasks'];tokens=env['encode'](z['frames'],enc)
    assert tokens.shape==(100,196,1024) and torch.isfinite(tokens).all()
    del enc;torch.cuda.empty_cache()
    save_npz(R/'inputs'/(panel+'_tokens.npz'),tokens=tokens.numpy(),tasks=np.array(spec['tasks']))
    model=env['SceneClassifier']().to(dev)
    model.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True));model.eval().requires_grad_(False)
    rows=[]
    with torch.no_grad():
        for i,task in enumerate(spec['tasks']):
            case=spec['cases'][task];path=Path(case['shortlist']);labels_path=path.with_name(path.stem+'_labels.npy')
            assert sha(path)==case['shortlist_sha256'] and sha(labels_path)==case['labels_sha256']
            with np.load(path) as z:actions=z['action'];indices=z['action_idx']
            status=np.load(labels_path);assert (status!=0).all();labels=status>0
            a=torch.from_numpy(actions).to(dev);tok=tokens[i:i+1].to(dev)
            scores=torch.cat([model(tok.expand(len(a[j:j+512]),-1,-1),a[j:j+512]) for j in range(0,len(a),512)]).cpu().numpy()
            assert scores.shape==(1024,) and np.isfinite(scores).all()
            order=np.argsort(-scores,kind='stable');hits=np.flatnonzero(labels[order]);first=int(hits[0]+1) if len(hits) else None
            # Separate repeated argmax audit of the nine actual selected trials.
            available=scores.copy();actual=None
            for turn in range(1,10):
                j=int(available.argmax());available[j]=-np.inf
                if labels[j]:actual=turn;break
            assert actual==(first if first is not None and first<=9 else None)
            folder=R/'evaluation'/panel/task.replace(':','_')
            save_npz(folder/'scores.npz',scores=scores,indices=indices,labels=labels,order=order)
            row={'task':task,'first_success':first,'provenance':prov,'checkpoint_sha256':spec['checkpoint_sha256']}
            write(folder/'result.json',row);rows.append(row)
            if (i+1)%10==0:print('EVAL',panel,i+1,flush=True)
    write(R/(panel+'_complete.json'),{'complete':True,'provenance':prov,'seconds':time.time()-start,
        'checkpoint_sha256':sha(spec['checkpoint']),'counts':{str(k):sum(r['first_success'] is not None and r['first_success']<=k for r in rows) for k in [1,3,5,9]},'tasks':rows})


def summary():
    plan=checked();prov=provenance();previous=read(M/'SUMMARY.json');out={'complete':True,'provenance':prov,'panels':{}}
    for panel,spec in plan['panels'].items():
        complete=read(R/(panel+'_complete.json'));assert complete['complete'] and complete['provenance']==prov
        cases={}
        for task in spec['tasks']:
            folder=R/'evaluation'/panel/task.replace(':','_');row=read(folder/'result.json');assert row['provenance']==prov
            with np.load(folder/'scores.npz') as z:
                order=np.argsort(-z['scores'],kind='stable');hits=np.flatnonzero(z['labels'][order]);first=int(hits[0]+1) if len(hits) else None
                np.testing.assert_array_equal(order,z['order'])
                with np.load(M/'evaluation'/panel/'q0'/task.replace(':','_')/'scores.npz') as base:
                    np.testing.assert_array_equal(z['indices'],base['indices']);np.testing.assert_array_equal(z['labels'],base['labels'])
            assert first==row['first_success'];cases[task]=first
        counts={f's{k}':sum(x is not None and x<=k for x in cases.values()) for k in [1,3,5,9]}
        assert counts=={'s'+k:v for k,v in complete['counts'].items()}
        arms={kind:{'counts':a['counts']} for kind,a in previous['panels'][panel].items()}
        arms['initial_scene']={'counts':counts,'first_success':cases,'seconds':complete['seconds'],'paired':{}}
        for name in ['base','q0']:
            refs=previous['panels'][panel][name]['first_success'];pairs={}
            for k in [1,3,5,9]:
                solved=lambda x:x is not None and x<=k
                pairs[k]={'gained':[t for t in cases if solved(cases[t]) and not solved(refs[t])],
                    'lost':[t for t in cases if not solved(cases[t]) and solved(refs[t])]}
            arms['initial_scene']['paired'][name]=pairs
        out['panels'][panel]=arms
    write(R/'SUMMARY.json',out);print(json.dumps({p:{k:a['counts'] for k,a in arms.items()} for p,arms in out['panels'].items()},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','render','eval','summary']);ap.add_argument('--panel',choices=['cross','within']);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='render':render()
    elif a.mode=='eval':evaluate(a.panel)
    else:summary()
