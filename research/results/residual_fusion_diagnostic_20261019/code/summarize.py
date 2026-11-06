"""Summarize only fully verified fixed-history intervention cases."""
import hashlib
import json
from pathlib import Path
from statistics import median
R=Path(__file__).resolve().parents[1]
plan=json.loads((R/'PLAN.json').read_text())
plan_sha=hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest()
out={'complete':True,'plan_sha256':plan_sha,'scope':'Fixed-history update sensitivity, not closed-loop task success','limits':plan['limits'],'panels':{}}
for panel,tasks in plan['tasks'].items():
    cases=[]
    for task in tasks:
        p=R/'cases'/panel/task.replace(':','_')/'result.json'
        if not p.exists():raise RuntimeError(f'Missing case {panel}/{task}')
        row=json.loads(p.read_text());assert row['complete'] and row['plan_sha256']==plan_sha
        for c in row['cases']:
            assert c['instrumentation_exact'] and c['original_before_after_scores_exact']
            cases.append(c)
    groups={}
    for group,rows in [('first',[x for x in cases if x['update_number']==1]),('later',[x for x in cases if x['update_number']>1])]:
        if not rows:continue
        components=[x['variants']['real']['components'] for x in rows]
        baseline={'n':len(rows),'median_component_rms':{k:median(c[k]['rms'] for c in components) for k in components[0] if isinstance(components[0][k],dict)},
            'median_residual_to_action_centered_rms':median(c['residual_signal']['centered_rms']/max(c['action_embedding']['centered_rms'],1e-12) for c in components),
            'median_residual_to_nonresidual_centered_rms':median(c['residual_signal']['centered_rms']/max(c['nonresidual_sum']['centered_rms'],1e-12) for c in components),
            'median_spatial_abs_retention':median(c['spatial_abs_retention'] for c in components),'variants':{}}
        for v in plan['variants']:
            vs=[x['variants'][v] for x in rows]
            auc=[x['ranking']['pair_auc']-r['before']['pair_auc'] for x,r in zip(vs,rows) if x['ranking']['pair_auc'] is not None and r['before']['pair_auc'] is not None]
            baseline['variants'][v]={'median_mixed_relative_change_vs_real':median(x['mixed_change_vs_real']['relative_l2'] for x in vs),
                'median_normalized_relative_change_vs_real':median(x['normalized_change_vs_real']['relative_l2'] for x in vs),
                'median_update_relative_change_vs_real':median(x['update_change_vs_real']['relative_l2'] for x in vs),
                'median_memory_relative_change_vs_before':median(x['memory_change_vs_before']['relative_l2'] for x in vs),
                'top1_changed_vs_real':sum(x['ranking']['top1_changed'] for x in vs),'median_top10_overlap_vs_real':median(x['ranking']['top10_overlap'] for x in vs),
                'next_action_successes':sum(x['ranking']['next_success'] for x in vs),
                'median_pair_auc_change_vs_before':median(auc) if auc else None,'pair_auc_defined_cases':len(auc)}
        groups[group]=baseline
    out['panels'][panel]={'tasks':len(tasks),'groups':groups}
(R/'SUMMARY.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,indent=2))
