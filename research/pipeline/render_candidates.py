import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
import phyre
from phyre import simulator as S

p = argparse.ArgumentParser()
p.add_argument('--data', default='/path/to/phyre-unc/data')
p.add_argument('--manifest', default='ball_within_template_f0_dev_n100')
p.add_argument('--pool', type=int, default=5000)
p.add_argument('--workers', type=int, default=24)
p.add_argument('--out', default='/mnt/meg/jjenchen/phyre-unc/candidates')
args = p.parse_args()
cache = phyre.get_default_100k_cache('ball')
ACT = cache.action_array


def work(task):
    path = os.path.join(args.out, args.manifest, task.replace(':', '_') + '.npz')
    if os.path.exists(path):
        return task, 0
    st = cache.load_simulation_states(task)[:args.pool]
    idx = np.flatnonzero(st != 0)
    sim = phyre.initialize_simulator([task], 'ball')
    frames, feats = [], []
    for i in idx:
        ui, ok = sim._action_mapper.action_to_user_input(ACT[i])
        scene = S.add_user_input_to_scene(sim._tasks[0].scene, ui, sim._keep_spaces)
        s0 = S.simulate_scene(scene, 1)[0]
        frames.append(S.scene_to_raster(s0).astype(np.uint8))
        feats.append(S.scene_to_featurized_objects(s0).features[0])
    np.savez_compressed(path, frames=np.stack(frames), feats0=np.stack(feats).astype(np.float32), action_idx=idx,
                        action=ACT[idx].astype(np.float32), status=st[idx])
    return task, len(idx)


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(os.path.join(args.out, args.manifest), exist_ok=True)
    tasks = sorted({json.loads(l)['task'] for l in open(os.path.join(args.data, 'manifests', args.manifest + '.jsonl'))})
    with Pool(args.workers) as pool:
        for k, (t, n) in enumerate(pool.imap_unordered(work, tasks)):
            if k % 10 == 0:
                print(k, t, n, f'{time.time() - t0:.0f}s', flush=True)
    print('done', len(tasks), f'{time.time() - t0:.0f}s')
