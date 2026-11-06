"""Post-selection distribution audit; never changes tasks, policies or targets."""
import argparse
import json
from pathlib import Path
import statistics as st
from redesign_lib import Store,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--old-data',required=True);a=p.parse_args()
    root=Path(a.root);old=json.loads(Path(a.old_data).read_text());store=Store(include_features=False)
    selection=json.loads((root/'selection.json').read_text());rows=[]
    for split in ['train_tasks','validation_tasks']:
        for t in selection[split]:
            tr=json.loads((root/'traces'/f'{t.replace(":","_")}.json').read_text());d=old['tasks'][t]
            old_support={str(store.files[i]) for i in d['support_pool']}
            old_union={str(store.files[i]) for i in d['support_pool']+d['query_pool']}
            def filename(r):return 'rollouts/'+t.replace(':','_')+'_'+str(r['action_idx']).zfill(6)+'.npz'
            bf=[r for r in tr['base']['attempts'] if r['status']<0]
            rows.append({'task':t,'split':split,'old_support_pool_size':len(old_support),'new_shortlist_size':tr['n_shortlist'],
                'old_base_first_attempt_success':d['base_trace']['attempts'][0]['solved'],
                'new_base_first_attempt_success':tr['base']['attempts'][0]['status']>0,
                'new_base_failed_attempts':len(bf),'new_random_failed_attempts':len(tr['random']['failures']),
                'base_failures_outside_old_support':sum(filename(r) not in old_support for r in bf),
                'base_failures_outside_old_union':sum(filename(r) not in old_union for r in bf),
                'query_successes':tr['query_successes'],'query_count':len(tr['query']),
                'matched_prefix_count':len(tr['episodes'])})
    summary={}
    for split in ['train_tasks','validation_tasks']:
        rr=[r for r in rows if r['split']==split];nfailed=sum(r['new_base_failed_attempts'] for r in rr)
        summary[split]={'n_scenes':len(rr),'old_support_pool_median':st.median(r['old_support_pool_size'] for r in rr),
            'old_base_first_attempt_success_rate':st.mean(r['old_base_first_attempt_success'] for r in rr),
            'new_base_first_attempt_success_rate':st.mean(r['new_base_first_attempt_success'] for r in rr),
            'new_base_failed_attempts':nfailed,'fraction_failed_attempts_outside_old_support':sum(r['base_failures_outside_old_support'] for r in rr)/nfailed if nfailed else None,
            'fraction_failed_attempts_outside_old_union':sum(r['base_failures_outside_old_union'] for r in rr)/nfailed if nfailed else None,
            'eligible_paired_scenes':sum(r['matched_prefix_count']>0 for r in rr),
            'query_scenes_with_success':sum(r['query_successes']>0 for r in rr),
            'query_positive_fraction':sum(r['query_successes'] for r in rr)/sum(r['query_count'] for r in rr)}
    write_json(root/'collection_analysis.json',{'scope':'descriptive selected-policy distribution change; old pool and new shortlist differ in selection as well as size; not a performance comparison',
        'summary':summary,'per_task':rows})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
