"""Collect bounded-reservoir traces before R0--R4 training."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from redesign_lib import Store, load_models, score_ids, write_json, digest, BASE, PROBE, OLD_MEMORY, PACKS


def partitions(store, task):
    rng = np.random.default_rng([20260928, *map(int, task.split(':'))])
    ids = rng.permutation(store.by[task])
    cut = len(ids) // 2
    return ids[:cut].tolist(), ids[cut:].tolist()


def replay(ids, initial_scores, store, rescore=None):
    scores = initial_scores.copy(); tried = np.zeros(len(ids), bool)
    failures, attempts = [], []
    for _ in range(min(4, len(ids))):
        j = int(np.argmax(np.where(tried, -np.inf, scores)))
        i = int(ids[j]); tried[j] = True
        # Only this selected outcome is exposed to the policy.
        success = bool(store.labels[i])
        attempts.append({'index': i, 'solved': success})
        if success: break
        failures.append(i)
        if rescore is not None and len(failures) < 4:
            scores = rescore(failures)
    return {'failures': failures, 'attempts': attempts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=4); p.add_argument('--merge', action='store_true')
    args = p.parse_args(); root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    store = Store()
    if args.merge:
        rows = {}
        for shard in range(args.nshards):
            d = json.loads((root / f'prep_{shard}.json').read_text())
            assert not rows.keys() & d['tasks'].keys()
            rows.update(d['tasks'])
        assert set(rows) == set(store.by)
        templates = sorted({t[:5] for t in rows})
        held = sorted(np.random.default_rng(20260928).choice(templates, 4, replace=False).tolist())
        train = sorted(t for t in rows if t[:5] not in held)
        validation = []
        for tp in held:
            ts = sorted(t for t in rows if t[:5] == tp)
            validation += np.random.default_rng(20260928 + int(tp)).choice(ts, min(16, len(ts)), replace=False).tolist()
        for t, d in rows.items():
            assert not set(d['support_pool']) & set(d['query_pool'])
            assert len(set(d['support_pool'] + d['query_pool'])) == len(store.by[t])
            for policy in ['base_trace', 'memory_trace']:
                assert all(not store.labels[i] for i in d[policy]['failures'])
                attempts = d[policy]['attempts']
                assert not any(a['solved'] for a in attempts[:-1])
        write_json(root / 'dataset.json', {
            'seed': 20260928, 'packs': PACKS, 'n_actions': len(store.refs),
            'files_sha256': __import__('hashlib').sha256('\n'.join(store.files).encode()).hexdigest(),
            'train_tasks': train, 'validation_tasks': sorted(validation), 'memory_holdout_templates': held,
            'holdout_scope': 'memory-only; frozen base and probe saw these training templates',
            'trace_scope': 'existing deduplicated encoded actions, support half only; not a 1024-action deployment pool',
            'checkpoint_hashes': {str(p): digest(p) for p in [BASE, PROBE, OLD_MEMORY]}, 'tasks': rows})
        print(json.dumps({'done': True, 'n_train_tasks': len(train), 'n_validation_tasks': len(validation),
                          'held_templates': held, 'n_actions': len(store.refs),
                          'nonempty_base_trace': sum(bool(d['base_trace']['failures']) for d in rows.values()),
                          'nonempty_memory_trace': sum(bool(d['memory_trace']['failures']) for d in rows.values())}), flush=True)
        return
    torch.set_num_threads(4)
    base, probe, rule, _, model = load_models(memory=OLD_MEMORY)
    path = root / f'prep_{args.shard}.json'
    result = json.loads(path.read_text()) if path.exists() else {'tasks': {}}
    t0 = time.time()
    for task in sorted(store.by)[args.shard::args.nshards]:
        if task in result['tasks']: continue
        support, queries = partitions(store, task)
        ids = support + queries
        scores = score_ids(base, probe, rule, store, ids)
        btrace = replay(support, scores[:len(support)], store)
        mtrace = replay(support, scores[:len(support)], store,
                        lambda ctx: score_ids(base, probe, rule, store, support, model, ctx))
        result['tasks'][task] = {'support_pool': support, 'query_pool': queries,
                                'base_scores': {str(i): float(s) for i, s in zip(ids, scores)},
                                'base_trace': btrace, 'memory_trace': mtrace}
        if len(result['tasks']) % 10 == 0:
            write_json(path, result)
            print(json.dumps({'shard': args.shard, 'tasks': len(result['tasks']), 'seconds': round(time.time()-t0)}), flush=True)
    write_json(path, result)
    print('PREP_DONE', args.shard, flush=True)


if __name__ == '__main__': main()
