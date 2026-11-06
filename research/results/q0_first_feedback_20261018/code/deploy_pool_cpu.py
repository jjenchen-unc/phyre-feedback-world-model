# Full100k feedback pilot: preserve official float64 simulation actions; predictor inputs remain float32.
"""CPU PHYRE worker for deploy-pool training; no model or ranking decisions."""
import json
import sys
from pathlib import Path
import numpy as np
import phyre
from phyre import simulator as S


def candidates(task, out):
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array[:5000]
    states = cache.load_simulation_states(task)[:5000]
    sim = phyre.initialize_simulator([task], 'ball')
    frames, features, indices = [], [], []
    for i in np.flatnonzero(states != 0):
        ui, ok = sim._action_mapper.action_to_user_input(actions[i])
        assert ok
        if S.check_for_occlusions(sim._serialized[0], ui, keep_space_around_bodies=sim._keep_spaces):
            continue
        scene = S.add_user_input_to_scene(sim._tasks[0].scene, ui, sim._keep_spaces)
        initial = S.simulate_scene(scene, 1)[0]
        frames.append(S.scene_to_raster(initial).astype(np.uint8))
        features.append(S.scene_to_featurized_objects(initial).features[0])
        indices.append(int(i))
    assert len(indices) >= 1024, (task, len(indices))
    np.savez_compressed(out, frames=np.stack(frames), features=np.stack(features).astype(np.float32),
                        action=actions[indices].astype(np.float32), action_idx=np.array(indices))
    return {'ok': True, 'count': len(indices)}


def serve():
    last = None
    sim = None
    for line in sys.stdin:
        req = json.loads(line)
        task = req['task']
        out = Path(req['out'])
        out.parent.mkdir(parents=True, exist_ok=True)
        if req['mode'] == 'candidates':
            raise RuntimeError('Legacy 5000 candidate mode disabled: use complete100k renderer')
        else:
            if task != last:
                sim = phyre.initialize_simulator([task], 'ball')
                last = task
            if req['mode'] == 'labels':
                actions = np.asarray(req['actions'], np.float64)
                labels = np.array([int(sim.simulate_action(0, a, need_images=False).status)
                                   for a in actions], dtype=np.int8)
                assert (labels != 0).all()
                np.save(out, labels)
                result = {'ok': True}
            else:
                assert req['mode'] == 'rollout'
                action = np.asarray(req['action'], np.float64)
                status = int(sim.simulate_action(0, action, need_images=False).status)
                assert status != 0
                ui, ok = sim._action_mapper.action_to_user_input(action)
                assert ok
                scene = S.add_user_input_to_scene(sim._tasks[0].scene, ui, sim._keep_spaces)
                states = S.simulate_scene(scene, 1017)[::8]
                frames = np.stack([S.scene_to_raster(s) for s in states]).astype(np.uint8)
                features = np.concatenate([S.scene_to_featurized_objects(s).features for s in states]).astype(np.float32)
                assert len(frames) == 128
                np.savez_compressed(out, frames=frames, features=features, action=action,
                                    status=status, task=task)
                result = {'ok': True, 'status': status}
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    serve()
