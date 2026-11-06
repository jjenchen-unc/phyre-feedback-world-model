"""Read-only independent audit of the frozen support-sampling data selection."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from common import surface_dist
from redesign_lib import Store, digest, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    root = a.root; m = json.loads((root/'manifest.json').read_text()); inv = json.loads((root/'inventory.json').read_text())
    assert digest(root/'inventory.json') == m['inventory_sha256']
    store = Store(); assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest() == m['store_files_sha256']
    assert ((store.actions >= 0) & (store.actions <= 1)).all()
    checked = 0; eligible = {'train': [], 'calibration': []}; action_counts = defaultdict(int)
    clusters = defaultdict(set)
    for path in m['source_hashes']:
        assert digest(path) == m['source_hashes'][path]
        if path.endswith('.jsonl'):
            for line in Path(path).read_text().splitlines():
                r = json.loads(line)
                if 'cluster' in r: clusters[(r['task'], r['cluster'])].add(r['file'])
    for row in inv['rows']:
        roles = [row[k] for k in ['discovery','support','query']]
        flattened = sum(roles, []); assert len(set(flattened)) == len(flattened)
        assert all(store.tasks[i] == row['task'] for i in flattened)
        assigned = defaultdict(set)
        for i in flattened:
            g = row['groups'][str(i)]; assigned[g['group']].add(g['role'])
        assert all(len(v) == 1 for v in assigned.values())
        role_by_file = {str(store.files[i]): role for role in ['discovery','support','query'] for i in row[role]}
        for (task, _), files in clusters.items():
            if task == row['task']:
                assert len({role_by_file[f] for f in files if f in role_by_file}) <= 1
        if not row['potential']: continue
        score = json.loads((root/'support_scores'/(row['task'].replace(':','_')+'.json')).read_text())
        assert score['indices'] == row['support'] and np.isfinite(score['scores']).all()
        # Python stable ordering, separately from producer's NumPy argsort.
        ordered = sorted(zip(score['indices'],score['scores']), key=lambda x:-x[1])
        high = {i for i, _ in ordered[:(len(ordered)+4)//5]}
        near, hard, far = [], [], []
        for i in row['support']:
            if store.labels[i]: continue
            distance = surface_dist(store.feature(i)); minimum = float(distance[1:].min())
            assert minimum == row['quality'][str(i)]['minimum']
            if minimum <= 5: near.append(i)
            if minimum >= 20:
                (hard if i in high else far).append(i)
        assert near == row['N']
        if near and hard and far and row['query']:
            eligible[row['split']].append(row['task'])
        target = next((r for r in m['rows'] if r['task'] == row['task']), None)
        if target:
            assert target['N'] == near and target['H'] == hard and target['F'] == far
            assert target['query'] == row['query'] and target['support'] == row['support']
            for cat in ['N','H','F','failures']: action_counts[cat] += len(target[cat])
        checked += 1
    assert sorted(eligible['train']) == sorted(r['task'] for r in m['rows'] if r['split'] == 'train')
    assert len(eligible['calibration']) == m['coverage']['calibration']['eligible_scenes']
    assert not {r['task'][:5] for r in m['rows']}&{r['task'][:5] for r in m['cross_rows']}
    # Verify full natural panel by independent cached scores and actual labels.
    with np.load(m['cross_pack']+'_meta.npz') as packed:
        tasks = packed['tasks']; labels = packed['solved'].astype(bool)
        for row in m['cross_rows']:
            ids = np.flatnonzero(tasks == row['task']).tolist(); assert ids == row['indices']
            path = Path(m['probe_run'])/'evaluation/cross'/row['task'].replace(':','_')/'predicted_mix_s0__base.npz'
            assert digest(path) == row['base_scores_sha256']
            with np.load(path) as saved: values = saved['log_score']
            first = ids[max(range(len(ids)), key=lambda j:values[j])]
            assert row['natural_context'] == ([] if labels[first] else [first])
            assert not set(row['support'])&set(row['query'])
    result = {'passed': True, 'manifest_sha256': digest(root/'manifest.json'), 'inventory_sha256': digest(root/'inventory.json'),
        'potential_scenes_checked': checked, 'train_scenes':len(eligible['train']), 'calibration_available':len(eligible['calibration']),
        'natural_cross_scenes_checked':len(m['cross_rows']), 'selected_support_counts':dict(action_counts),
        'checks':['whole-cluster role separation','normalized actions','geometry categories recomputed',
                  'top20 stable rank including successes','common training-scene intersection',
                  'train/cross template separation','natural first-action failure only'],
        'auditor_sha256':digest(__file__)}
    write_json(root/'DATA_AUDIT.json', result); print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
