"""Render requested official-cache actions without any learned preselection."""
import json
import sys
from pathlib import Path
import numpy as np
import phyre
from phyre import simulator as S

cache = phyre.get_default_100k_cache('ball')
last, sim = None, None
for line in sys.stdin:
    req = json.loads(line)
    task = req['task']
    if task != last:
        sim = phyre.initialize_simulator([task], 'ball')
        states = cache.load_simulation_states(task)
        last = task
    frames, features, ids = [], [], []
    for idx in req['indices']:
        # Only validity is used here; success labels cannot change selection.
        if states[idx] == 0:
            continue
        ui, ok = sim._action_mapper.action_to_user_input(cache.action_array[idx])
        if not ok or S.check_for_occlusions(sim._serialized[0], ui, keep_space_around_bodies=sim._keep_spaces):
            continue
        scene = S.add_user_input_to_scene(sim._tasks[0].scene, ui, sim._keep_spaces)
        frame = S.simulate_scene(scene, 1)[0]
        frames.append(S.scene_to_raster(frame).astype(np.uint8))
        features.append(S.scene_to_featurized_objects(frame).features[0])
        ids.append(idx)
    path = Path(req['out'])
    np.savez(path, indices=np.asarray(ids, np.int64),
             frames=np.asarray(frames, np.uint8), features=np.asarray(features, np.float32))
    print(json.dumps({'ok': True, 'count': len(ids)}), flush=True)
