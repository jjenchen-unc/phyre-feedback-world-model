from study import *
from evaluate import rank


def main(seed=None):
    plan=checked();seed=plan['seed'] if seed is None else seed;reports={};verified=0
    receipts=[read(R/'models'/f'seed{seed}'/arm/'complete.json') for arm in ARMS]
    assert len({r['initial_sha'] for r in receipts})==1 and len({r['reader_sha'] for r in receipts})==1 and len({r['query_schedule_sha'] for r in receipts})==1
    for stage in ['discovery','confirmation']:
        rows={};raw={}
        for task in plan[stage]:
            folder=R/'evaluation'/f'seed{seed}'/stage/task.replace(':','_');row=read(folder/'result.json');assert row['complete'] and row['reader_unchanged'] and row['plan']==sha(R/'PLAN.json')
            with np.load(folder/'scores.npz') as z:
                for name,expected in row['records'].items():
                    actual=rank(z[name],z['labels'],[int(z['first'])] if row['first_failed'] else [])
                    assert actual==expected,(task,name,actual,expected);verified+=1
                raw[task]={k:z[k].copy() for k in z.files}
            rows[task]=row
        eligible=[t for t,r in rows.items() if r['first_failed']];summary={};paired={}
        for name in ['q0']+ARMS:
            ms=[rows[t]['records'][name] for t in eligible]
            summary[name]={k:float(np.mean([m[k] for m in ms if m[k] is not None])) if any(m[k] is not None for m in ms) else None for k in ['top1','top3','top5','top8','auc']}
            # Population total includes the identical already-successful first actions.
            summary[name]['success_first_or_next1']=(len(rows)-len(eligible)+sum(m['top1'] for m in ms))/len(rows)
            summary[name]['success_first_or_next8_static']=(len(rows)-len(eligible)+sum(m['top8'] for m in ms))/len(rows)
        for arm in ARMS:
            control='prediction_'+arm.split('_')[1];zero=arm.split('_')[0]+'_zero'
            paired[arm]={}
            for comp in sorted(set(['q0',control,zero])):
                ds={k:[rows[t]['records'][arm][k]-rows[t]['records'][comp][k] for t in eligible if rows[t]['records'][arm][k] is not None and rows[t]['records'][comp][k] is not None] for k in ['top1','top3','top5','top8','auc']}
                paired[arm][comp]={k:dict(mean=float(np.mean(d)),gain=sum(v>1e-12 for v in d),loss=sum(v< -1e-12 for v in d),tie=sum(abs(v)<=1e-12 for v in d),n=len(d)) for k,d in ds.items()}
        losses={arm:{k:float(np.mean([rows[t]['losses'][arm][k] for t in eligible])) for k in ['prediction','l1','geometry','bce','pair']} for arm in ARMS}
        reports[stage]=dict(n=len(rows),first_failed=len(eligible),summary=summary,paired=paired,losses=losses)
    candidates={}
    for loss in ['bce','pair']:
        arm=loss+'_residual';deltas={stage:{comp:{k:reports[stage]['paired'][arm][comp][k]['mean'] for k in ['top1','top3','top5','top8']} for comp in ['prediction_residual',loss+'_zero','q0']} for stage in reports}
        promising=all(deltas[s]['prediction_residual']['top1']>=0 and deltas[s]['prediction_residual']['top8']>=0 for s in reports) and sum(deltas[s]['prediction_residual']['top8'] for s in reports)>0
        content=all(deltas[s][loss+'_zero']['top1']>=0 and deltas[s][loss+'_zero']['top8']>=0 for s in reports) and sum(deltas[s][loss+'_zero']['top8'] for s in reports)>0
        candidates[loss]=dict(promising_vs_prediction=promising,residual_content_gate=content,deltas=deltas)
    write(R/f'ROUND1_SUMMARY_seed{seed}.json',dict(complete=True,plan=sha(R/'PLAN.json'),seed=seed,raw_vectors_verified=verified,matched_initialization_and_query_schedule=True,training=receipts,reports=reports,candidates=candidates,closed_loop=False,decision='Human audit required before adaptive second round; this screening is not a significance test'))
    print('ROUND1_COMPLETE',candidates,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);a=p.parse_args();main(a.seed)
