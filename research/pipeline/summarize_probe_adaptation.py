"""Verify raw probe-adaptation outputs and aggregate scene-level paired effects."""
import argparse
import json
from pathlib import Path
import numpy as np
from attribution_a_metrics import log_score,auc
from redesign_lib import write_json,digest


def mean(v):
    v=[x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();root=a.root
    manifest=json.loads((root/'manifest.json').read_text());stage=json.loads((Path(manifest['stage_a'])/'manifest.json').read_text())
    expected=[('cross',r['task']) for r in stage['tasks'] if not a.smoke or r['task'] in manifest['smoke_cross_tasks']]
    if not a.smoke:expected += [('calibration',r['task']) for r in manifest['rows'] if r['split']=='calibration']
    folder=root/('smoke_eval' if a.smoke else 'evaluation');rows=[];array_count=0
    code={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}
    for panel,task in expected:
        d=folder/panel/task.replace(':','_');row=json.loads((d/'summary.json').read_text());assert row['task']==task and row['panel']==panel
        assert row['provenance']['manifest']==digest(root/'manifest.json') and row['provenance']['code']==code
        assert row['input_boundary_checked'] and row['empty_memory_identity']
        if panel=='cross':assert sorted(row['original_exact_replay'])==['base','memory','observed']
        with np.load(d/'truth.npz') as t:
            mask=t['query_mask'];y=t['labels'];td=t['distance'];assert t['indices'][mask].tolist()==row['query']
            for key,metrics in row['conditions'].items():
                with np.load(d/(key+'.npz')) as v:
                    np.testing.assert_allclose(log_score(v['distance'],row['rule']),v['log_score'],rtol=1e-12,atol=1e-12)
                    assert auc(y[mask],v['log_score'][mask])==metrics['auc']
                    for group,gm in [('all',mask),('success',mask&y),('failure',mask&~y)]:
                        for metric,value in [('latent_l1',v['latent_l1']),('surface_mae',np.abs(v['distance'][:,1:]-td[:,1:]).mean(1))]:
                            expected_mean=float(value[gm].mean()) if gm.any() else None;actual=metrics[group][metric]
                            assert (actual is None and expected_mean is None) or abs(actual-expected_mean)<1e-7,(task,key,group,metric)
                    original=row['conditions']['original__'+key.split('__')[1]]
                    assert metrics['all']['latent_l1']==original['all']['latent_l1'],'Probe changed frozen latent errors'
                    array_count+=1
        rows.append(row)
    if a.smoke:
        write_json(root/'smoke_check.json',{'passed':True,'tasks':len(rows),'arrays_verified':array_count,'scope':'Engineering only; no model selection','code':code});print('SMOKE_PASSED',len(rows),array_count);return
    keys=['auc','hit_at_10','first_success_rank','all.latent_l1','all.surface_mae','success.surface_mae','failure.surface_mae','success.green_position_error','all.green_position_error','success.near_contact_tau_time_disagreement']
    def get(metrics,key):
        for k in key.split('.'):metrics=metrics[k]
        return metrics
    panels={'cross_nonempty_K1_primary':[r for r in rows if r['panel']=='cross' and r['context']],
            'cross_all100':[r for r in rows if r['panel']=='cross'],
            'calibration':[r for r in rows if r['panel']=='calibration']}
    result={'complete':True,'audit_passed':True,'arrays_verified':array_count,'scope':manifest['scope'],'panels':{}}
    for name,rr in panels.items():
        conditions={}
        for condition in rr[0]['conditions']:
            conditions[condition]={'n_scenes':len(rr),'macro':{},'metric_scene_counts':{}}
            for key in keys:
                values=[get(r['conditions'][condition],key) for r in rr];conditions[condition]['macro'][key]=mean(values);conditions[condition]['metric_scene_counts'][key]=sum(v is not None for v in values)
        paired={}
        for seed in manifest['seeds']:
            mix=f'predicted_mix_s{seed}';control=f'true_control_s{seed}';initial=f'initial_only_s{seed}';changes=[]
            for r in rr:
                c=r['conditions'];entry={'task':r['task'],'template':r['task'][:5]}
                for metric in ['auc','success.surface_mae','all.surface_mae']:
                    vals={k:get(c[k],metric) for k in [mix+'__memory',control+'__memory','original__memory',mix+'__base','original__base',initial+'__initial',mix+'__initial']}
                    if any(v is None for v in vals.values()):
                        entry[metric]=None;continue
                    entry[metric]={'memory_mix_minus_original':vals[mix+'__memory']-vals['original__memory'],
                                   'memory_mix_minus_true_control':vals[mix+'__memory']-vals[control+'__memory'],
                                   'memory_base_gap_change':(vals[mix+'__memory']-vals[mix+'__base'])-(vals['original__memory']-vals['original__base']),
                                   'memory_mix_minus_trained_initial':vals[mix+'__memory']-vals[initial+'__initial'],
                                   'mix_memory_minus_initial_ablation':vals[mix+'__memory']-vals[mix+'__initial']}
                changes.append(entry)
            aggregate={}
            for metric in ['auc','success.surface_mae','all.surface_mae']:
                good=[v for v in changes if v[metric] is not None];aggregate[metric]={'n_scenes':len(good),'mean':{},'by_template':{}}
                if good:
                    for comparison in good[0][metric]:
                        aggregate[metric]['mean'][comparison]=mean([v[metric][comparison] for v in good])
                    for template in sorted(set(v['template'] for v in good)):
                        tr=[v for v in good if v['template']==template]
                        aggregate[metric]['by_template'][template]={'n_scenes':len(tr),**{comparison:mean([v[metric][comparison] for v in tr]) for comparison in good[0][metric]}}
            paired[str(seed)]={'aggregate':aggregate,'per_task':changes}
        result['panels'][name]={'conditions':conditions,'paired':paired}
    write_json(root/'SUMMARY.json',result)
    c=result['panels']['cross_nonempty_K1_primary']['conditions'];lines=['# Probe adaptation results','','Reused cross D0; frozen base and R0; common K1/query; fixed final checkpoints.','AUC and success geometry use AUC/success-defined scenes; all denominators in SUMMARY.json.','', '| Probe | Input | AUC | Success surface MAE (px) |','|---|---|---:|---:|']
    for key,data in c.items():
        head,source=key.split('__');m=data['macro'];lines.append(f'| {head} | {source} | {m["auc"]:.5f} | {m["success.surface_mae"]:.3f} |')
    lines+=['','Three probe seeds share the same pretrained head/base/memory. No new-template population or closed-loop success claim.','A positive readout intervention supports a remediable readout bottleneck; it does not uniquely prove latent distribution shift or physical correctness.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'complete':True,'tasks':len(rows),'arrays_verified':array_count}),flush=True)

if __name__=='__main__':main()
