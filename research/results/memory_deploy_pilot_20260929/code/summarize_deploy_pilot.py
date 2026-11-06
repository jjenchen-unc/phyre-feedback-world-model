"""Describe train resubstitution, collection, and paired pilot results separately."""
import argparse
import json
from pathlib import Path
import statistics as st
from redesign_lib import write_json


def fixed_stats(d):
    rows=d['rows'];out={}
    for c in sorted({r['condition'] for r in rows}):
        rr=[r for r in rows if r['condition']==c and r.get('actual_k',0)>0 and r.get('requested_k',1)==1]
        # Each scene has equal weight; prefix counts must not reweight tasks.
        values=[]
        for task in sorted({r['task'] for r in rr}):
            rs=[r for r in rr if r['task']==task];auc=[r['rank_auc'] for r in rs if r['rank_auc'] is not None]
            values.append({'task':task,'auc':st.mean(auc) if auc else None,
                'l1':st.mean(q['latent_l1'] for r in rs for q in r['per_query'])})
        out[c]={'n_tasks':len(values),'rank_auc':st.mean(v['auc'] for v in values if v['auc'] is not None) if any(v['auc'] is not None for v in values) else None,
                'latent_l1':st.mean(v['l1'] for v in values) if values else None,'per_task':values}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--runs',required=True);a=p.parse_args()
    root=Path(a.root);runs=Path(a.runs);result={'train_diagnostic':{},'pilot':{}};missing=[]
    for v in ['R0','R1','R2','R3','R4']:
        f=root/f'{v}_train_fixed.json'
        if f.exists():result['train_diagnostic'][v]=fixed_stats(json.loads(f.read_text()))
        else:missing.append(f.name)
    for name in ['collection_summary','pairing_audit']:
        f=root/(name+'.json')
        if f.exists():result[name]=json.loads(f.read_text())
        else:missing.append(f.name)
    ready=result.get('pairing_audit',{}).get('ab_ready',False)
    initial=root/'initial_R0_fixed.json'
    if ready:
        if initial.exists():result['initial_R0']=fixed_stats(json.loads(initial.read_text()))
        else:missing.append(initial.name)
    for v in ['A','B']:
        for seed in [0,1]:
            key=f'{v}_s{seed}';f=root/f'{key}_fixed.json';complete=runs/key/'training_complete.json'
            if ready:
                if f.exists() and complete.exists():
                    result['pilot'][key]=fixed_stats(json.loads(f.read_text()))
                    result.setdefault('training_counts',{})[key]=json.loads(complete.read_text())['counts']
                else:missing.append(key)
    if len(result['pilot'])==4:
        result['paired_policy_context_delta_B_minus_A']={}
        for seed in [0,1]:
            assert result['training_counts'][f'A_s{seed}']==result['training_counts'][f'B_s{seed}']
            aa={r['task']:r['auc'] for r in result['pilot'][f'A_s{seed}']['policy_context']['per_task']}
            bb={r['task']:r['auc'] for r in result['pilot'][f'B_s{seed}']['policy_context']['per_task']}
            assert aa.keys()==bb.keys()
            delta={t:bb[t]-aa[t] for t in aa if aa[t] is not None and bb[t] is not None}
            result['paired_policy_context_delta_B_minus_A'][str(seed)]={'per_task':delta,'mean':st.mean(delta.values()) if delta else None}
    result['complete']=not missing;result['missing']=missing;result['ab_ready']=ready
    write_json(root/'SUMMARY.json',result)
    lines=['# Deployment memory pilot','',f'Complete: {result["complete"]}; A/B eligible: {ready}',
        '', 'Training diagnostics are resubstitution; A/B is fixed-query, memory-only held-template validation.',
        'Two fine-tuning seeds share one pretrained R0 initialization. No full adaptive cross evaluation.',
        '', '| Model | Train nonempty scenes | Base AUC | Correct AUC | Wrong-scene AUC |','|---|---:|---:|---:|---:|']
    for v,r in result['train_diagnostic'].items():
        lines.append(f'| {v} | {r["base"]["n_tasks"]} | {r["base"]["rank_auc"]:.4f} | {r["correct"]["rank_auc"]:.4f} | {r["wrong_scene"]["rank_auc"]:.4f} |')
    lines+=['','| Model | Pilot policy-context AUC | Random-context AUC |','|---|---:|---:|']
    if 'initial_R0' in result:
        r=result['initial_R0'];lines.append(f'| unchanged R0 | {r["policy_context"]["rank_auc"]} | {r["random_context"]["rank_auc"]} |')
    for key,r in result['pilot'].items():lines.append(f'| {key} | {r["policy_context"]["rank_auc"]} | {r["random_context"]["rank_auc"]} |')
    if not ready:lines+=['','A/B skipped or not ready; inspect pairing_audit.json.']
    if missing:lines+=['','Missing: '+', '.join(missing)]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'complete':not missing,'missing':missing,'ab_ready':ready}),flush=True)
    if missing:raise SystemExit(2)


if __name__=='__main__':main()
