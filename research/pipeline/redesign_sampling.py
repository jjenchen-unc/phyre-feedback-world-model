"""Deterministic paired episode sampling, independent of model predictions."""
import numpy as np


def sample_episode(data, store, step, seed, variant):
    rng = np.random.default_rng([seed, step])
    intended = float(rng.random())
    source = 'base_trace' if .4 <= intended < .7 else 'memory_trace' if intended >= .7 else None
    if '_eligible' not in data:
        data['_eligible'] = {p: [t for t in data['train_tasks'] if data['tasks'][t][p]['failures']]
                             for p in ['base_trace', 'memory_trace']}
    eligible = data['_eligible'][source] if source else data['train_tasks']
    if not eligible: eligible = data['train_tasks']
    # All variants share this conditional scene schedule, including R0/R2.
    task = eligible[int(rng.integers(len(eligible)))]
    d = data['tasks'][task]
    pool = d['support_pool']; qp = d['query_pool']
    k = int(rng.choice([1, 2, 4]))
    k = min(k, len(pool))
    kind, fallback = 'mixed', None
    if intended < .1:
        context = []; kind = 'empty'
    elif variant in ['R0', 'R2']:
        context = rng.choice(pool, k, replace=False).tolist()
    else:
        fail = [i for i in pool if not store.labels[i]]
        if intended >= .4:
            kind = 'base_trace' if intended < .7 else 'memory_trace'
            context = d[kind]['failures'][:k]
            if not context:
                fallback = 'policy_had_no_failed_prefix'
        else:
            kind = 'diverse_failure'; context = []
        if not context:
            context = rng.choice(fail, min(k, len(fail)), replace=False).tolist() if fail else []
    # Query draws must not depend on how many random numbers context used.
    rng = np.random.default_rng([seed, step, 1])
    if variant in ['R0', 'R1']:
        query = rng.choice(qp, min(8, len(qp)), replace=False).tolist()
        groups = ['uniform'] * len(query)
    else:
        query, groups = [], []
        pos = [i for i in qp if store.labels[i]]
        neg = [i for i in qp if not store.labels[i]]
        def add(candidates, label):
            take = [i for i in candidates if i not in query][:2]
            query.extend(take); groups.extend([label] * len(take))
        # Diverse successes via farthest-point traversal from a seeded anchor.
        diverse = []
        if pos:
            diverse = [int(rng.choice(pos))]
            while len(diverse) < min(2, len(pos)):
                others = [i for i in pos if i not in diverse]
                distances = np.linalg.norm(store.actions[others, None] - store.actions[diverse][None], axis=-1).min(1)
                diverse.append(others[int(distances.argmax())])
        add(diverse, 'positive')
        add(sorted(neg, key=lambda i: -d['base_scores'][str(i)]), 'high_score_failure')
        if pos:
            dist = {i: float(np.linalg.norm(store.actions[pos] - store.actions[i], axis=-1).min()) for i in neg}
            add(sorted(neg, key=lambda i: dist[i]), 'near_success_failure')
        add(rng.permutation(qp).tolist(), 'uniform')
        for i in rng.permutation(qp):
            if len(query) >= min(8, len(qp)): break
            if int(i) not in query: query.append(int(i)); groups.append('shortage_fallback')
    assert not set(context) & set(query)
    assert len(query) == len(set(query)) and len(context) == len(set(context))
    if variant not in ['R0', 'R2']: assert not store.labels[context].any()
    return {'task': task, 'context': context, 'query': query, 'groups': groups,
            'context_kind': kind, 'context_fallback': fallback, 'requested_k': k}
