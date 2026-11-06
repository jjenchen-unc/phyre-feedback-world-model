"""CPU cost of validity/initial renders and true labels on bounded cached actions."""
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import phyre
from phyre import simulator as S

ROOT=Path('/path/to/phyre-unc')


def case(task):
    start=time.perf_counter()
    cache=phyre.get_default_100k_cache('ball')
    actions=cache.action_array
    states=cache.load_simulation_states(task)
    sim=phyre.initialize_simulator([task],'ball')
    startup=time.perf_counter()-start
    chosen=[];begin=time.perf_counter()
    for i in np.flatnonzero(states[:5000]!=0):
        ui,ok=sim._action_mapper.action_to_user_input(actions[i])
        if not ok or S.check_for_occlusions(sim._serialized[0],ui,keep_space_around_bodies=sim._keep_spaces):continue
        scene=S.add_user_input_to_scene(sim._tasks[0].scene,ui,sim._keep_spaces)
        state=S.simulate_scene(scene,1)[0]
        frame=S.scene_to_raster(state)
        feature=S.scene_to_featurized_objects(state).features[0]
        assert frame.shape==(256,256) and feature.shape[-1]==14
        chosen.append(int(i))
        if len(chosen)==64:break
    render=time.perf_counter()-begin;begin=time.perf_counter()
    labels=[int(sim.simulate_action(0,actions[i],need_images=False).status) for i in chosen]
    assert len(chosen)==64 and all(v!=0 for v in labels)
    return {'task':task,'actions':64,'startup_s':startup,'render_valid_s':render,'labels_s':time.perf_counter()-begin,
            'total_s':time.perf_counter()-start}


if __name__=='__main__':
    out=ROOT/'results/multiturn_validation_20261014/cpu_cost.json'
    assert not out.exists()
    tasks=[]
    for panel,fold in [('cross',0),('cross',2),('within',1),('within',2)]:
        rows=[json.loads(x) for x in (ROOT/'data/manifests'/('ball_'+panel+'_template_f'+str(fold)+'_train.jsonl')).read_text().splitlines()]
        available=sorted({r['task'] for r in rows})
        tasks.extend(available[int(i)] for i in np.linspace(0,len(available)-1,8,dtype=int))
    result={'complete':False,'scope':'32 sampled scenes x64 actions, no scientific outcome evaluation','serial':[],'parallel':[]}
    start=time.perf_counter()
    for task in tasks[:8]:result['serial'].append(case(task))
    result['serial_wall_s']=time.perf_counter()-start
    out.write_text(json.dumps(result,indent=2)+'\n')
    start=time.perf_counter()
    with Pool(4) as pool:
        for record in pool.imap(case,tasks[8:]):
            result['parallel'].append(record);out.write_text(json.dumps(result,indent=2)+'\n')
    result['parallel_workers']=4;result['parallel_wall_s']=time.perf_counter()-start;result['complete']=True
    out.write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',result['serial_wall_s'],result['parallel_wall_s'])
