"""Generate unexposed tasks from existing templates; no model-outcome selection."""
import argparse
from collections import Counter
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
import phyre
import phyre.loader
import phyre.eval_task_complexity as complexity
from phyre import simulator as S
from phyre.interface.task.ttypes import Task
from thrift import TSerialization

PROJECT = Path('/path/to/phyre-unc')
PARENT = PROJECT / 'results/support_utility_20261004'
SEED = 20261004


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)
def task_hash(task):
    return hashlib.sha256(S.serialize(task.scene) + repr((task.bodyId1, task.bodyId2, task.relationships)).encode()).hexdigest()
def load_task(root, task_id):
    return TSerialization.deserialize(Task(), (root/'tasks'/(task_id.replace(':','_')+'.bin')).read_bytes())


def select(root, assets):
    assert not (root/'selection.json').exists()
    parent = read(PARENT/'manifest.json'); templates = sorted({r['task'][:5] for r in parent['rows'] if r['split']=='train'})
    compiled = phyre.loader.load_compiled_task_dict(); existing_hashes = {task_hash(t) for t in compiled.values()}
    exposed = set(); sources = {}
    for path in sorted((PROJECT/'data/manifests').glob('*.jsonl')):
        sources[str(path)] = sha(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'task' in row: exposed.add(row['task'])
    chosen = []; counts = {}; task_hashes = set(); t0 = time.time()
    for tp in templates:
        _, source, module = phyre.loader.load_task_script(tp); builder = module.build_task
        stats = complexity.maybe_load_evaluation(tp); assert stats is not None
        generated = builder.build_tasks_for_search(tp); eligible = []; matched = 0
        for task in generated:
            h = task_hash(task)
            if task.taskId in compiled:
                assert h == task_hash(compiled[task.taskId]), ('generator/compiled mismatch', task.taskId)
                matched += 1
            if task.taskId in compiled or task.taskId in exposed or h in existing_hashes: continue
            if not builder._check_flags(stats['flags'], task.taskId): continue
            eligible.append(task)
        rng = np.random.default_rng([SEED, int(tp), 941])
        order = rng.permutation(len(eligible)); selected = []
        for idx in order:
            task = eligible[int(idx)]; h = task_hash(task)
            if h in task_hashes: continue
            task_hashes.add(h); selected.append(task)
            if len(selected)==10: break
        assert len(selected)==10, (tp, 'insufficient fresh curated scenes', len(selected))
        sources[str(source)] = sha(source)
        for path in [complexity.get_evaluation_path(source), complexity.get_evaluation_meta_path(source)]: sources[path] = sha(path)
        counts[tp] = {'generated_search_scenes':len(generated),'matched_compiled':matched,'eligible_fresh_same_flags':len(eligible),'selected':len(selected),
            'required_flags':builder.search_params.required_flags,'excluded_flags':builder.search_params.excluded_flags,
            'diversify_tier':builder.search_params.diversify_tier}
        for task in selected:
            path = root/'tasks'/(task.taskId.replace(':','_')+'.bin'); path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(TSerialization.serialize(task))
            chosen.append({'task':task.taskId,'template':tp,'parameters':{k:float(v) for k,v in task.template_params.items()},
                'scene_goal_sha256':task_hash(task),'task_file_sha256':sha(path)})
        print(tp, json.dumps(counts[tp]), flush=True)
    actions = phyre.get_default_100k_cache('ball').action_array[:5000].astype(np.float32)
    assets.mkdir(parents=True,exist_ok=True); np.save(assets/'actions.npy', actions)
    tasks = sorted(r['task'] for r in chosen)
    selected = {'scope':'Fresh uncompiled instances of ten seen templates, same published solvability flags; not official within fold',
        'tasks':tasks,'rows':chosen,'templates':templates,'counts':counts,'exposure_sources':sources,
        'excluded_compiled_count':len(compiled),'excluded_recorded_count':len(exposed),
        'parent':str(PARENT),'parent_manifest_sha256':sha(PARENT/'manifest.json'),
        'assets':str(assets),'action_array_sha256':sha(assets/'actions.npy'),'seed':SEED,
        'initial_candidate_budget':5000,'shortlist':1024,
        'selection_rule':'uniform seeded sampling among uncompiled, unrecorded, nonduplicate scene-goal pairs satisfying existing template flags',
        'limitation':'Template diversity subset selection is not rerun; within/cross differ in template mix and scene-selection procedure.',
        'seconds':time.time()-t0}
    write(root/'selection.json',selected)
    write(root/'EXPOSURE_AUDIT.json',{'passed':True,'selection_sha256':sha(root/'selection.json'),'new_scenes':len(chosen),
        'templates':templates,'all_ids_absent_from_compiled_and_recorded':True,'all_scene_goal_hashes_new':True,'sources':sources})
    print('SELECTED',len(tasks),flush=True)


def candidate_one(item):
    root, assets, task_id, smoke = item; root=Path(root); assets=Path(assets)
    folder=assets/('smoke_candidates' if smoke else 'candidates'); folder.mkdir(parents=True,exist_ok=True)
    path=folder/(task_id.replace(':','_')+'.npz')
    if path.exists(): return task_id, 'existing'
    task=load_task(root,task_id); sim=phyre.ActionSimulator([task],'ball')
    actions=np.load(assets/'actions.npy')[:128 if smoke else 5000]; ids=[]; frames=[]; feats=[]
    for i, action in enumerate(actions):
        ui, valid=sim._action_mapper.action_to_user_input(action)
        if not valid: continue
        if S.check_for_occlusions(sim._serialized[0],ui,keep_space_around_bodies=sim._keep_spaces):continue
        scene=S.add_user_input_to_scene(task.scene,ui,sim._keep_spaces)
        s0=S.simulate_scene(scene,1)[0]
        frames.append(S.scene_to_raster(s0).astype(np.uint8));feats.append(S.scene_to_featurized_objects(s0).features[0]);ids.append(i)
    # No rollout outcomes are computed or fed into shortlist selection.
    assert len(ids)>=(8 if smoke else 1024),(task_id,len(ids))
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,frames=np.stack(frames),feats0=np.stack(feats).astype(np.float32),
        action_idx=np.array(ids),action=actions[ids],task_sha256=sha(root/'tasks'/(task_id.replace(':','_')+'.bin')))
    tmp.replace(path);return task_id,len(ids)


def rollout_one(item):
    root,assets,task_id,smoke=item;root=Path(root);assets=Path(assets)
    scores=assets/('smoke_scores' if smoke else 'scores')/(task_id.replace(':','_')+'.npz')
    data=assets/('smoke_data' if smoke else 'data');raw=data/'rollouts';raw.mkdir(parents=True,exist_ok=True)
    result_path=data/'task_manifests'/(task_id.replace(':','_')+'.json')
    if result_path.exists():return read(result_path)
    task=load_task(root,task_id);sim=phyre.ActionSimulator([task],'ball');rows=[]
    with np.load(scores) as p:actions=p['action'];aids=p['action_idx']
    assert len(actions)==(8 if smoke else 1024)
    for action,ai in zip(actions,aids):
        ai=int(ai);path=raw/f'{task_id.replace(":","_")}_{ai:06d}.npz'
        if path.exists():
            with np.load(path) as p: status=int(p['status']);assert np.array_equal(p['action'],action) and len(p['frames'])==128
        else:
            status=int(sim.simulate_action(0,action,need_images=False).status);assert status!=0
            ui,valid=sim._action_mapper.action_to_user_input(action);assert valid
            scene=S.add_user_input_to_scene(task.scene,ui,sim._keep_spaces)
            states=S.simulate_scene(scene,1017)[::8]
            frames=np.stack([S.scene_to_raster(s) for s in states]).astype(np.uint8)
            features=np.concatenate([S.scene_to_featurized_objects(s).features for s in states]).astype(np.float32)
            assert len(frames)==128
            tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,frames=frames,features=features,action=action,
                action_idx=ai,status=status,steps=1017,stride=8,task=task_id);tmp.replace(path)
        rows.append({'task':task_id,'action_idx':ai,'status':status,'solved':status>0,'file':'rollouts/'+path.name})
    write(result_path,rows);return rows


def run_collection(root,mode,smoke,workers):
    sel=read(root/'selection.json');assets=Path(sel['assets']);tasks=sel['tasks']
    if smoke:tasks=[next(t for t in tasks if t.startswith(tp+':')) for tp in [sel['templates'][0],sel['templates'][-1]]]
    if mode=='candidates':
        with Pool(workers) as pool:
            for value in pool.imap_unordered(candidate_one,[(str(root),str(assets),t,smoke) for t in tasks]):print(value,flush=True)
    else:
        rows=[]
        with Pool(workers) as pool:
            for value in pool.imap_unordered(rollout_one,[(str(root),str(assets),t,smoke) for t in tasks]):
                rows.extend(value);print('ROLLOUT',value[0]['task'],len(value),flush=True)
        rows.sort(key=lambda r:(r['task'],r['action_idx']))
        data=assets/('smoke_data' if smoke else 'data');(data/'manifests').mkdir(exist_ok=True)
        path=data/'manifests/within_support.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(root/(('smoke_' if smoke else '')+mode+'_complete.json'),{'complete':True,'selection_sha256':sha(root/'selection.json'),'tasks':tasks})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','candidates','rollouts']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--assets',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=12)
    a=p.parse_args()
    if a.stage=='select':select(a.root,a.assets)
    else:run_collection(a.root,a.stage,a.smoke,a.workers)


if __name__=='__main__':main()
