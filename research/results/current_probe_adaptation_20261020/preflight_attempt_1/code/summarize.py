from adapt_runtime import *
import argparse
import numpy as np


def mean_task(rows, field):
    # Equal scene, then available variant, then depth weights.
    per_variant = {}
    for row in rows:
        value = field(row)
        if value is not None:
            per_variant.setdefault(row['variant'], []).append(value)
    return float(np.mean([np.mean(v) for v in per_variant.values()])) if per_variant else None


def audit_heads(smoke=False):
    root = R/('smoke_heads' if smoke else 'heads')
    rows = [read(root/a/'complete.json') for a in ARMS[1:]]
    assert all(r['complete'] and r['plan'] == sha(R/'PLAN.json') for r in rows)
    assert len({r['draw_sha256'] for r in rows}) == len({r['initial_sha256'] for r in rows}) == 1
    for arm, row in zip(ARMS[1:], rows):
        assert sha(root/arm/'probe.pt') == row['checkpoint_sha256']
    return rows


def decide():
    plan = checked()
    audit_heads()
    tasks = {}
    for task in plan['validation']:
        r = read(R/'fixed'/task.replace(':', '_')/'result.json')
        assert r['complete'] and r['old_scores_exact'] and r['plan'] == sha(R/'PLAN.json')
        tasks[task] = r['rows']
    summary = {}
    scene_metrics = {}
    for arm in ARMS:
        scene_metrics[arm] = {
            t: {k: mean_task(rows, fun) for k, fun in {
                'auc': lambda r: r['heads'][arm]['auc'],
                'top1': lambda r: r['heads'][arm]['top1'],
                'top8': lambda r: r['heads'][arm]['top8'],
                'predicted_geometry': lambda r: r['heads'][arm]['predicted']['loss'],
                'observed_geometry': lambda r: r['heads'][arm]['observed']['loss'],
            }.items()} for t, rows in tasks.items()}
        summary[arm] = {k: float(np.mean([r[k] for r in scene_metrics[arm].values() if r[k] is not None]))
                        for k in next(iter(scene_metrics[arm].values()))}
    candidate = summary['mixed']
    passed = all(candidate['auc'] > summary[a]['auc']
                 and candidate['predicted_geometry'] < summary[a]['predicted_geometry']
                 and candidate['top8'] >= summary[a]['top8'] for a in ['old', 'true_control'])
    # If no jointly definable AUC scenes exist, never pass the gate.
    assert all(np.isfinite(v) for m in summary.values() for v in m.values())
    write(R/'DECISION.json', dict(complete=True, plan=sha(R/'PLAN.json'), run_closed_loop=passed,
                                 summary=summary, scene_metrics=scene_metrics,
                                 n_scenes=len(tasks), n_auc_scenes=sum(r['auc'] is not None for r in scene_metrics['old'].values()),
                                 interpretation='Exploratory readout gate, not statistical proof. Fixed latents/supports isolate probe effects; conditional later-state population.'))
    print('DECISION', passed, summary, flush=True)


def final():
    plan = checked()
    decision = read(R/'DECISION.json')
    groups = {}
    if decision['run_closed_loop']:
        for arm in ARMS[1:]:
            for first in ['own', 'common']:
                for variant in VARIANTS:
                    result = {}
                    for task in plan['test']:
                        dest = R/'closed'/arm/first/variant/task.replace(':', '_')
                        r = read(dest/'result.json')
                        assert r['complete'] and r['model_unchanged'] and r['plan'] == sha(R/'PLAN.json')
                        with np.load(dest/'scores.npz') as z:
                            scores, selected, labels = z['scores'], z['selected'], z['labels']
                            assert np.isfinite(scores).all() and len(set(selected)) == len(selected) == len(scores)
                            for turn, j in enumerate(selected):
                                v = scores[turn].copy()
                                v[selected[:turn]] = -np.inf
                                expected = r['common_first'] if first == 'common' and turn == 0 else int(v.argmax())
                                assert j == expected
                            hits = np.flatnonzero(labels[selected])
                            actual = int(hits[0]+1) if len(hits) else None
                            assert actual == r['first_success'] and len(selected) == (actual or 9)
                        result[task] = actual
                    groups[f'{arm}/{first}/{variant}'] = dict(
                        counts={f's{k}': sum(v is not None and v <= k for v in result.values()) for k in [1, 3, 5, 9]},
                        first_success=result)
    write(R/'FINAL_SUMMARY.json', dict(complete=True, plan=sha(R/'PLAN.json'), decision=decision, closed_loop=groups))
    print('COMPLETE', len(groups), 'closed-loop groups', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['smoke', 'heads', 'decision', 'final'])
    a = p.parse_args()
    if a.stage in ['smoke', 'heads']:
        audit_heads(a.stage == 'smoke')
        if a.stage == 'smoke':
            assert read(R/'PREFLIGHT.json')['passed']
            assert read(R/'ENCODER_VALIDATION.json')['passed']
        write(R/('SMOKE.json' if a.stage == 'smoke' else 'HEAD_AUDIT.json'), dict(passed=True))
    elif a.stage == 'decision':
        decide()
    else:
        final()
