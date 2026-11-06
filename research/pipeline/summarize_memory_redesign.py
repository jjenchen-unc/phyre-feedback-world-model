"""Aggregate completed R0--R4 jobs; never label partial jobs complete."""
import argparse
import json
from pathlib import Path
import statistics
from redesign_lib import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--runs',required=True)
    args=p.parse_args();out=Path(args.out);runs=Path(args.runs)
    summary={}
    for name in ['R0','R1','R2','R3','R4']:
        complete=runs/name/'training_complete.json';cl=out/(name+'_closed_loop.json');fixed=out/(name+'_fixed.json')
        row={'training_complete':complete.exists(),'closed_loop_complete':cl.exists(),'fixed_complete':fixed.exists()}
        if complete.exists():row['training']=json.loads(complete.read_text())
        if cl.exists():row['closed_loop']=json.loads(cl.read_text())['aggregate']
        if fixed.exists():
            d=json.loads(fixed.read_text());result={}
            for k in [1,4]:
                for condition in ['base','empty','correct','wrong_scene']:
                    rs=[r for r in d['rows'] if r['requested_k']==k and r['condition']==condition]
                    result[f'k{k}_{condition}']={
                        'n_tasks':len(rs),'n_nonempty_context_tasks':sum(r['actual_k']>0 for r in rs),
                        'rank_auc':statistics.mean(r['rank_auc'] for r in rs if r['rank_auc'] is not None),
                        'success_latent_l1':statistics.mean(q['latent_l1'] for r in rs for q in r['per_query'] if q['solved']),
                        'failure_latent_l1':statistics.mean(q['latent_l1'] for r in rs for q in r['per_query'] if not q['solved'])}
            row['fixed']=result
        summary[name]=row
    done=all(r['training_complete'] and r['closed_loop_complete'] and r['fixed_complete'] for r in summary.values())
    write_json(out/'SUMMARY.json',{'complete':done,'variants':summary})
    lines=['# Memory redesign first-wave results','',f'Complete: {done}',
           '','Single seed, memory-only internal holdout, reused cross-f1 development panel; not final test evidence.',
           '','| Variant | Training | Cross dev AUCCESS | S@10 | S@100 |','|---|---|---:|---:|---:|']
    for n,r in summary.items():
        c=r.get('closed_loop',{});s=c.get('success_at',{})
        lines.append(f'| {n} | {r["training_complete"]} | {c.get("auccess","pending")} | {s.get("10","pending")} | {s.get("100","pending")} |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':done,'summary':str(out/'SUMMARY.json')}),flush=True)
    if not done: raise SystemExit(2)


if __name__=='__main__':main()
