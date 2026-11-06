"""Check exported per-scene metric arithmetic and all three latest CSV tables.

Uses only the standard library. This is not a raw-score or simulator replay.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'research/results'
TOP = ['top1', 'top3', 'top5', 'top8']


def read(path):
    return json.loads(path.read_text())


def aggregate(panel, nested=False):
    rows = panel['per_scene']
    if nested:
        rows = {t: r['metrics'] for t, r in rows.items() if r['failed']}
    assert len(rows) == panel['first_failed'] == 97
    assert panel['n'] == 100
    result = {}
    for name in next(iter(rows.values())):
        values = [r[name] for r in rows.values()]
        counts = {k: sum(v[k] for v in values) for k in TOP}
        auc = [v['auc'] for v in values if v['auc'] is not None]
        result[name] = dict(counts, auc=sum(auc)/len(auc), S2_count=3+counts['top1'])
        reported = panel['summary'][name]
        for k in TOP:
            assert counts[k] == (reported['counts'][k] if nested else reported[k]), (name, k)
        assert abs(result[name]['auc'] - reported['auc']) < 1e-12
        assert result[name]['S2_count'] == reported['two_attempt_success_count' if nested else 'S2_count']
    return result


def check_csv(path, scores):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len({r['method'] for r in rows}) == len(rows)
    for row in rows:
        expected = scores[row['method']]
        for k in TOP + ['S2_count']:
            assert int(row[k]) == expected[k], (path, row['method'], k)
        assert abs(float(row['auc']) - expected['auc']) < 1e-12
    return len(rows)


def main():
    root = ROOT / 'success_objective_loop_20261021'
    earlier = read(root/'INDEPENDENT_ANALYSIS.json')
    assert earlier['complete']
    scores = aggregate(earlier['panels']['cross100'], nested=True)
    contact = read(root/'FINAL_INDEPENDENT_AUDIT.json')['contact']['panels']['cross100']
    for name, val in contact['summary'].items():
        rows = contact['per_scene'].values()
        counts = {k: sum(r[name][k] for r in rows) for k in TOP}
        for k in TOP:
            assert counts[k] == val[k]
        scores[name] = dict(val, S2_count=3+counts['top1'])
    n = check_csv(root/'CROSS100_FINAL_TABLE.csv', scores)
    print('Success-objective comparison:', n, 'CSV rows verified')
    for experiment in ['feedback_readout_loop_20261021', 'support_context_ablation_20261021']:
        root = ROOT / experiment
        audit = read(root/'AUDIT.json')
        assert audit['complete']
        scores = aggregate(audit['panels']['cross100'])
        n = check_csv(root/'CROSS100_TABLE.csv', scores)
        print('\n' + experiment + ':', n, 'CSV rows verified')
        for name, val in sorted(scores.items()):
            if 'shuffle' not in name:
                print(f"{name:32} S@2={val['S2_count']:2}/100  next1/3/5/8="
                      + '/'.join(str(val[k]) for k in TOP) + ' (97 first failures)')
    print('\nPublished per-scene metric counts agree with CSVs. No raw-score replay performed.')


if __name__ == '__main__':
    main()
