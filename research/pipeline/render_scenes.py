import argparse, json, os
import numpy as np
import phyre
from phyre import simulator as S

p = argparse.ArgumentParser()
p.add_argument('--setup', default='ball_within_template')
p.add_argument('--fold', type=int, default=0)
p.add_argument('--out', default='/mnt/meg/jjenchen/phyre-unc/scenes')
args = p.parse_args()
train, dev, _ = phyre.get_fold(args.setup, args.fold)
cache = phyre.get_default_100k_cache('ball')
os.makedirs(args.out, exist_ok=True)
for split, tasks in [('train', train), ('dev', dev)]:
    sim = phyre.initialize_simulator(tasks, 'ball')
    frames = np.stack([S.scene_to_raster(t.scene).astype(np.uint8) for t in sim._tasks])
    status = np.stack([cache.load_simulation_states(t) for t in tasks]).astype(np.int8)
    np.savez_compressed(os.path.join(args.out, f'{args.setup}_f{args.fold}_{split}.npz'), frames=frames, status=status,
                        tasks=np.array(tasks), actions=cache.action_array.astype(np.float32))
    print(split, frames.shape, status.shape, 'solved frac %.4f' % (status > 0).mean(), 'valid frac %.3f' % (status != 0).mean())
