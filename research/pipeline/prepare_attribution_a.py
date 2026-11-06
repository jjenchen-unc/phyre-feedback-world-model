"""Freeze the existing D0 panel and common base-policy contexts before new inference."""
import argparse
import json
from pathlib import Path

import numpy as np
from redesign_lib import BASE, PROBE, ROOT, PROJECT, digest, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    pack = ROOT / 'packs/cross_f1_dev_shortlist1024_pca128'
    source = PROJECT / 'results/stable_scorer_cross_20260930'
    baseline = json.loads((source / 'base_stable_log.json').read_text())
    with np.load(str(pack) + '_meta.npz') as m:
        tasks, labels, files, actions = [m[k] for k in ['tasks', 'solved', 'files', 'action']]
    models = {'initial_R0': str(ROOT / 'runs/memory_redesign_20260928/R0/memory.pt')}
    models.update({n: str(ROOT / 'runs/memory_deploy_pilot_20260929' / n / 'memory.pt') for n in ['A_s0', 'B_s0', 'A_s1', 'B_s1']})
    train_templates = set()
    for name in ['cross_f1_train_clusters_pca128', 'cross_f1_train_s16r16_pca128']:
        with np.load(ROOT / 'packs' / (name + '_meta.npz')) as m:
            train_templates.update(str(t).split(':')[0] for t in m['tasks'])
    rows = []; trace_hashes = {}
    for r in baseline['per_task']:
        task = r['task']; ids = np.flatnonzero(tasks == task)
        assert len(ids) == 1024 and len(set(str(x) for x in files[ids])) == 1024
        path = Path(str(source / 'base_stable_log.json') + '.tasks') / (task.replace(':', '_') + '.json')
        trace = json.loads(path.read_text()); trace_hashes[task] = digest(path)
        failures = []
        for step in trace['attempts']:
            i = step['pack_index']; assert i in ids and bool(labels[i]) == step['solved']
            assert np.array_equal(actions[i], step['action'])
            if step['solved']: break
            failures.append(i)
        ctx = {'K1': failures[:1]}
        if len(failures) >= 4: ctx['K4'] = failures[:4]
        excluded = set(failures[:4])  # Same remaining queries for K1/K4 and every model.
        query = [int(i) for i in ids if int(i) not in excluded]
        assert all(not labels[i] for i in excluded)
        rows.append({'task': task, 'indices': ids.tolist(), 'query': query, 'contexts': ctx,
                     'base_first_success': r['first_success'], 'solutions_in_pool': int(labels[ids].sum()),
                     'has_nonempty_K1': bool(ctx['K1']), 'exposure': 'reused_cross_development_D0'})
    assert len(rows) == 100
    test_templates = sorted({r['task'].split(':')[0] for r in rows})
    assert not set(test_templates) & train_templates
    smoke = sum([[r['task'] for r in sorted(rows, key=lambda r: r['task']) if r['task'].startswith(t + ':')][:2] for t in test_templates], [])
    manifest = {'scope': 'Stage A only, existing D0; no D1/D2 sampling or confirmation claims.',
                'pack': str(pack), 'meta_sha256': digest(str(pack) + '_meta.npz'),
                'base': str(BASE), 'base_sha256': digest(BASE), 'probe': str(PROBE), 'probe_sha256': digest(PROBE),
                'models': {n: {'path': p, 'sha256': digest(p)} for n, p in models.items()},
                'base_trace_source': str(source), 'base_trace_hashes': trace_hashes,
                'training_pack_templates': sorted(train_templates), 'evaluation_templates': test_templates,
                'smoke_tasks': smoke, 'tasks': rows,
                'counts': {'tasks': len(rows), 'nonempty_K1': sum(r['has_nonempty_K1'] for r in rows),
                           'K4': sum('K4' in r['contexts'] for r in rows),
                           'K1_with_solutions': sum(r['has_nonempty_K1'] and r['solutions_in_pool'] > 0 for r in rows)}}
    write_json(a.out / 'manifest.json', manifest)
    print(json.dumps(manifest['counts']), flush=True)


if __name__ == '__main__': main()
