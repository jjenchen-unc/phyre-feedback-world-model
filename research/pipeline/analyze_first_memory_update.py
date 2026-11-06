"""Locate first-update ranking damage in saved stable closed-loop traces."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    base_file = root / 'base_stable_log.json'
    base = {r['task']: r for r in json.loads(base_file.read_text())['per_task']}
    result = {'scope': 'Post-hoc stable-scorer trace analysis; same first failure and common remaining queries. Conditional on a failed first attempt and a solution in the shortlist.', 'models': {}}
    for name in ['initial_R0', 'A_s0', 'B_s0', 'A_s1', 'B_s1']:
        path = root / (name + '_stable_log.json')
        rows = []
        for r in json.loads(path.read_text())['per_task']:
            stem = r['task'].replace(':', '_')
            td = Path(str(path) + '.tasks')
            trace = json.loads((td / (stem + '.json')).read_text())
            bt = json.loads((Path(str(base_file) + '.tasks') / (stem + '.json')).read_text())
            with np.load(td / trace['score_file']) as z, np.load(Path(str(base_file) + '.tasks') / bt['score_file']) as bz:
                scores, labels, ids = z['scores'][:, 1], z['labels'], z['indices']
                assert np.array_equal(ids, bz['indices']) and np.array_equal(labels, bz['labels'])
                assert np.array_equal(scores[0], bz['scores'][0, 1])
                assert trace['attempts'][0]['pack_index'] == bt['attempts'][0]['pack_index']
                if len(scores) < 2 or not labels.any():
                    continue
                first = trace['attempts'][0]['candidate_local']
                assert not labels[first] and trace['contexts'][1] == [int(ids[first])]
                eligible = np.arange(len(labels)) != first
                def rank(s):
                    order = np.argsort(-s[eligible], kind='stable')
                    return int(np.flatnonzero(labels[eligible][order])[0]) + 1
                before, after = rank(scores[0]), rank(scores[1])
                rows.append({'task': r['task'], 'base_rank_after_excluding_first_failure': before,
                             'memory_rank_after_one_failure': after, 'rank_delta': after - before,
                             'base_first_success': base[r['task']]['first_success'],
                             'memory_first_success': r['first_success'],
                             'base_solved_memory_failed': bool(base[r['task']]['first_success']) and not bool(r['first_success'])})
        assert len(rows) == 57
        result['models'][name] = {'n_tasks': len(rows),
            'rank_improved': sum(r['rank_delta'] < 0 for r in rows),
            'rank_unchanged': sum(r['rank_delta'] == 0 for r in rows),
            'rank_worsened': sum(r['rank_delta'] > 0 for r in rows),
            'median_rank_delta': float(np.median([r['rank_delta'] for r in rows])),
            'per_task': rows}
    (root / 'FIRST_UPDATE_ANALYSIS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'per_task'} for k, v in result['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
