import argparse, json, os, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import phyre
from phyre import simulator as S

p = argparse.ArgumentParser()
p.add_argument('--setup', default='ball_within_template')
p.add_argument('--fold', type=int, default=0)
p.add_argument('--split', choices=['train', 'dev', 'test'], default='train')
p.add_argument('--max-tasks', type=int, default=0)
p.add_argument('--n-success', type=int, default=2)
p.add_argument('--n-fail', type=int, default=2)
p.add_argument('--steps', type=int, default=1017)
p.add_argument('--stride', type=int, default=8)
p.add_argument('--verify', type=int, default=20)
p.add_argument('--workers', type=int, default=8)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default='/path/to/phyre-unc/data')
p.add_argument('--tasks-from', default='')
p.add_argument('--tag', default='')
p.add_argument('--diverse', action='store_true')
p.add_argument('--n-random', type=int, default=0)
p.add_argument('--clusters', type=int, default=0)
p.add_argument('--actions-file', default='')
p.add_argument('--cluster-size', type=int, default=8)
p.add_argument('--radius', type=float, default=0.08)
args = p.parse_args()

cache = phyre.get_default_100k_cache('ball')
ACTIONS = cache.action_array


def pick_tasks():
    if args.actions_file:
        return sorted(ACTION_LIST)
    if args.tasks_from:
        return sorted({json.loads(l)['task'] for l in open(os.path.join(args.out, 'manifests', args.tasks_from + '.jsonl'))})
    train, dev, test = phyre.get_fold(args.setup, args.fold)
    tasks = {'train': train, 'dev': dev, 'test': test}[args.split]
    if not args.max_tasks or args.max_tasks >= len(tasks):
        return list(tasks)
    by_t = defaultdict(list)
    for t in tasks:
        by_t[t.split(':')[0]].append(t)
    rng = np.random.default_rng(args.seed)
    per = int(np.ceil(args.max_tasks / len(by_t)))
    out = [t for tp in sorted(by_t) for t in rng.choice(by_t[tp], min(per, len(by_t[tp])), replace=False)]
    return sorted(out)[:args.max_tasks] if len(out) > args.max_tasks else sorted(out)


def pick_clusters(task, st, rng):
    valid = np.flatnonzero(st != 0)
    sol, fail = valid[st[valid] > 0], valid[st[valid] < 0]
    if len(sol) == 0:
        return []
    anchors = list(rng.choice(sol, min(args.clusters // 2, len(sol)), replace=False))
    for s0 in rng.choice(sol, args.clusters - len(anchors), replace=True):
        d = np.linalg.norm(ACTIONS[fail] - ACTIONS[s0], axis=1)
        anchors.append(fail[int(d.argmin())])
    out, used = [], set()
    for c, a0 in enumerate(anchors):
        d = np.linalg.norm(ACTIONS[valid] - ACTIONS[a0], axis=1)
        near = [i for i in valid[np.argsort(d)] if i != a0 and i not in used and np.linalg.norm(ACTIONS[i] - ACTIONS[a0]) < args.radius]
        ns, nf = [i for i in near if st[i] > 0], [i for i in near if st[i] < 0]
        k = args.cluster_size - 1
        take_s = list(rng.permutation(ns)[:k // 2])
        take_f = list(rng.permutation(nf)[:k - len(take_s)])
        take_s += list(rng.permutation([i for i in ns if i not in take_s])[:k - len(take_s) - len(take_f)])
        for i in [a0] + take_s + take_f:
            used.add(int(i))
            out.append((int(i), int(st[i]), c))
    return out


ACTION_LIST = {}
if args.actions_file:
    for l in open(args.actions_file):
        r = json.loads(l)
        ACTION_LIST.setdefault(r['task'], []).append(int(r['action_idx']))


def pick_actions(task):
    st = cache.load_simulation_states(task)
    rng = np.random.default_rng([args.seed, *map(int, task.split(':'))])
    if args.actions_file:
        return [(i, int(st[i])) for i in ACTION_LIST.get(task, [])]
    if args.clusters:
        return pick_clusters(task, st, rng)
    out = []
    for want, n in ((1, args.n_success), (-1, args.n_fail)):
        idx = np.flatnonzero(st == want)
        if args.diverse and len(idx) > n:
            a = ACTIONS[idx]
            chosen = [int(rng.integers(len(idx)))]
            d = np.linalg.norm(a - a[chosen[0]], axis=1)
            while len(chosen) < n:
                chosen.append(int(d.argmax()))
                d = np.minimum(d, np.linalg.norm(a - a[chosen[-1]], axis=1))
            out += [(int(idx[c]), want) for c in chosen]
        else:
            out += [(int(i), want) for i in rng.choice(idx, min(n, len(idx)), replace=False)]
    if args.n_random:
        taken = {i for i, _ in out}
        idx = np.array([i for i in np.flatnonzero(st != 0) if i not in taken])
        out += [(int(i), int(st[i])) for i in rng.choice(idx, min(args.n_random, len(idx)), replace=False)]
    return out


def work(item):
    task, verify = item
    sim = phyre.initialize_simulator([task], 'ball')
    raw = os.path.join(args.out, 'rollouts')
    rows = []
    for pick in pick_actions(task):
        ai, cache_status = pick[0], pick[1]
        path = os.path.join(raw, f'{task.replace(":", "_")}_{ai:06d}.npz')
        a = ACTIONS[ai]
        status = int(sim.simulate_action(0, a, need_images=False).status)
        row = {'task': task, 'action_idx': ai, 'cache_status': cache_status, 'status': status,
               'solved': status > 0, 'file': os.path.relpath(path, args.out)}
        if len(pick) > 2:
            row['cluster'] = pick[2]
        if not os.path.exists(path):
            ui, ok = sim._action_mapper.action_to_user_input(a)
            assert ok
            scene = S.add_user_input_to_scene(sim._tasks[0].scene, ui, sim._keep_spaces)
            scenes = S.simulate_scene(scene, args.steps)[::args.stride]
            frames = np.stack([S.scene_to_raster(sc) for sc in scenes]).astype(np.uint8)
            feats = np.concatenate([S.scene_to_featurized_objects(sc).features for sc in scenes]).astype(np.float32)
            tmp = path + '.tmp.npz'
            np.savez_compressed(tmp, frames=frames, features=feats, action=a.astype(np.float32),
                                action_idx=ai, status=status, cache_status=cache_status,
                                steps=args.steps, stride=args.stride, task=task)
            os.replace(tmp, path)
        else:
            z = np.load(path)
            frames, feats = z['frames'], z['features']
        row['num_frames'] = int(frames.shape[0])
        row['bytes'] = os.path.getsize(path)
        if verify:
            ref = sim.simulate_action(0, a, need_images=True, need_featurized_objects=True, stride=args.stride)
            n = len(ref.images)
            row['verify_prefix_frames'] = n
            row['verify_images_equal'] = bool(np.array_equal(frames[:n], ref.images))
            row['verify_features_maxdiff'] = float(np.abs(feats[:n] - ref.featurized_objects.features).max())
        rows.append(row)
    return rows


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(os.path.join(args.out, 'rollouts'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'manifests'), exist_ok=True)
    tasks = pick_tasks()
    items = [(t, i < args.verify) for i, t in enumerate(tasks)]
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.imap(work, items, chunksize=4) for r in rs]
    name = f'{args.setup}_f{args.fold}_{args.split}' + (f'_n{args.max_tasks}' if args.max_tasks else '') + args.tag
    with open(os.path.join(args.out, 'manifests', name + '.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    ver = [r for r in rows if 'verify_images_equal' in r]
    summary = {'manifest': name, 'tasks': len(tasks), 'rollouts': len(rows),
               'solved': sum(r['solved'] for r in rows), 'failed': sum(not r['solved'] for r in rows),
               'tasks_short_of_successes': sum(1 for t in tasks if sum(r['task'] == t and r['cache_status'] == 1 for r in rows) < args.n_success),
               'label_mismatch_vs_cache': sum((r['status'] > 0) != (r['cache_status'] > 0) for r in rows),
               'frame_counts': sorted({r['num_frames'] for r in rows}),
               'verified': len(ver), 'verify_all_images_equal': all(r['verify_images_equal'] for r in ver),
               'verify_max_feature_diff': max((r['verify_features_maxdiff'] for r in ver), default=None),
               'mean_mb_per_rollout': round(np.mean([r['bytes'] for r in rows]) / 1e6, 3),
               'total_gb': round(sum(r['bytes'] for r in rows) / 1e9, 3),
               'seconds': round(time.time() - t0, 1), 'args': vars(args)}
    with open(os.path.join(args.out, 'manifests', name + '.summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
