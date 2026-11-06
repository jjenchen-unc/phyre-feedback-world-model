from direction_runtime import *
import argparse


def aggregate(stage):
    plan=checked(); branch=read(R/'ROUND1.json'); chosen=branch['selected_variant']; zero=chosen.split('_')[0]+'_zero'
    results={}; replayed=0
    for task in plan[stage]:
        d=R/stage/task.replace(':','_'); rec=read(d/'result.json')
        assert rec['complete'] and rec['old_exact'] and rec['model_unchanged'] and rec['plan']==sha(R/'PLAN.json')
        results[task]=rec
        for row in rec['records']:
            with np.load(d/(row['variant']+'.npz')) as z:
                for name,values in row['heads'].items():
                    actual=rank_metrics(z[name],z['labels'],z['tried'])
                    for k,v in actual.items():
                        expected=values['rank'][k]
                        assert (v is None and expected is None) or (v is not None and expected is not None and abs(v-expected)<1e-13)
                    if rec['first_failed']:
                        actual=rank_metrics(z[name],z['labels'],[int(z['qfirst'])])
                        for k,v in actual.items():
                            expected=values['after_first_failure'][k]
                            assert (v is None and expected is None) or (v is not None and expected is not None and abs(v-expected)<1e-13)
                    replayed+=1
    summary={}; paired={}
    for variant in ['q0',chosen,zero]:
        rows={t:next(x for x in rec['records'] if x['variant']==variant) for t,rec in results.items() if any(x['variant']==variant for x in rec['records'])}
        assert rows
        names=list(next(iter(rows.values()))['heads'])
        summary[variant]={'n':len(rows),'heads':{}}
        for name in names:
            metrics=mean_metrics([row['heads'][name]['rank'] for row in rows.values()])
            geometry={s:{role:float(np.mean([row['heads'][name]['geometry'][s][role] for row in rows.values() if row['heads'][name]['geometry'][s][role] is not None])) for role in ['all','task','closest_sampled_time']} for s in ['predicted','observed']}
            summary[variant]['heads'][name]=dict(rank=metrics,geometry=geometry)
        paired[variant]={}
        for name in names:
            if name in ['old','previous_mixed']:continue
            diff={k:[row['heads'][name]['rank'][k]-row['heads']['old']['rank'][k] for row in rows.values() if row['heads']['old']['rank'][k] is not None] for k in ['top1','top8','auc']}
            paired[variant][name]={k:dict(mean=float(np.mean(v)),positive=sum(x>1e-12 for x in v),negative=sum(x< -1e-12 for x in v),zero=sum(abs(x)<=1e-12 for x in v),n=len(v)) for k,v in diff.items()}
    # Same exact first-failure population and same tried action across all three states.
    transfer={}
    eligible=[t for t,r in results.items() if r['first_failed']]
    qrows={t:next(x for x in results[t]['records'] if x['variant']=='q0') for t in eligible}
    mrows={t:next(x for x in results[t]['records'] if x['variant']==chosen) for t in eligible}
    zrows={t:next(x for x in results[t]['records'] if x['variant']==zero) for t in eligible}
    for name in mrows[eligible[0]]['heads']:
        mm=mean_metrics([mrows[t]['heads'][name]['rank'] for t in eligible]); zz=mean_metrics([zrows[t]['heads'][name]['rank'] for t in eligible])
        # Original Q0 reader comparator plus matched domain-specific Q0 reader for each seed.
        qname='q0_mixed_'+name.rsplit('_',1)[1] if name not in ['old','previous_mixed'] else name
        qq=mean_metrics([qrows[t]['heads'][qname]['after_first_failure'] for t in eligible])
        qo=mean_metrics([qrows[t]['heads']['old']['after_first_failure'] for t in eligible])
        transfer[name]=dict(n=len(eligible),memory=mm,zero=zz,q0_domain=qq,q0_old=qo,
                            delta_vs_zero={k:mm[k]-zz[k] if mm[k] is not None else None for k in mm},
                            delta_vs_q0_old={k:mm[k]-qo[k] if mm[k] is not None else None for k in mm},
                            delta_vs_q0_domain={k:mm[k]-qq[k] if mm[k] is not None else None for k in mm})
    return dict(stage=stage,n=len(results),eligible_first_failures=len(eligible),summary=summary,paired=paired,transfer=transfer,raw_rankings_verified=replayed)


def arm_quality(report,variant,arm):
    heads=report['summary'][variant]['heads']; old=heads['old']['rank']; values=[heads[arm+'_'+str(s)]['rank'] for s in SEEDS]
    delta8=[v['top8']-old['top8'] for v in values];delta1=[v['top1']-old['top1'] for v in values]
    passed=min(delta8)>=-1e-12 and np.mean(delta8)>1e-12 and min(delta1)>=-1e-12
    return dict(pass_gate=bool(passed),top8_delta=delta8,top1_delta=delta1,
                key=[min(delta8),float(np.mean(delta8)),float(np.mean(delta1)),float(np.mean([v['auc']-old['auc'] for v in values]))])


def decision():
    checked(); report=aggregate('discovery'); variant=read(R/'ROUND1.json')['selected_variant']
    qualities={arm:arm_quality(report,variant,arm) for arm in ARMS[1:]}
    eligible=[a for a,q in qualities.items() if q['pass_gate']]
    selected=max(eligible or ARMS[1:],key=lambda arm:qualities[arm]['key'])
    value=dict(complete=True,plan=sha(R/'PLAN.json'),selected_memory_arm=selected,
               discovery_promising=qualities[selected]['pass_gate'],qualities=qualities,
               q0_quality=arm_quality(report,'q0','q0_mixed'),report=report,
               selection_scope='Discovery16 only; confirmation16 not loaded. Confirmation still runs if no candidate passes, to evaluate stability of the best observed direction.')
    write(R/'ROUND2_DECISION.json',value)
    print('ROUND2_DECISION',selected,qualities,flush=True)


def final():
    plan=checked(); earlier=read(R/'ROUND2_DECISION.json'); report=aggregate('confirmation');variant=read(R/'ROUND1.json')['selected_variant'];arm=earlier['selected_memory_arm']
    quality=arm_quality(report,variant,arm);q0quality=arm_quality(report,'q0','q0_mixed')
    names=[arm+'_'+str(s) for s in SEEDS];transfer=[report['transfer'][n] for n in names]
    content_gate=all(t['delta_vs_zero']['top8']>=-1e-12 and t['delta_vs_zero']['top1']>=-1e-12 for t in transfer) and np.mean([t['delta_vs_zero']['top8'] for t in transfer])>1e-12
    reader_replicated=earlier['discovery_promising'] and quality['pass_gate']
    initial_replicated=earlier['q0_quality']['pass_gate'] and q0quality['pass_gate']
    if reader_replicated and content_gate:
        direction='Evaluate the selected memory readout in a small closed loop with the same-first-action and zero-residual controls before any larger updater training.'
        reason='Early-rank gain versus old reader repeats across both scene partitions and both probe samplingseeds, and the same-history residual state beats zero in confirmation.'
    elif reader_replicated:
        direction='Prioritize success-oriented readout calibration; keep residual-content attribution unresolved and retain action-only/zero controls in the next closed-loop pilot.'
        reason='Readout improvement repeats, but a specific advantage of residual content over zero is not established.'
    elif initial_replicated:
        direction='Validate Q0-specific readout in closed loop first; investigate updater success-sensitive supervision separately before scaling memory training.'
        reason='Initial prediction readout improves consistently, while the tested memory-readout intervention does not meet replication criteria.'
    else:
        direction='Prioritize a small updater experiment with success-sensitive supervision on matched action pairs, retaining no-update and zero-residual controls; do not scale geometry-only probe adaptation.'
        reason='None of these bounded readout changes shows repeatable early-rank gains. This does not prove latent information loss or exclude other probe designs.'
    value=dict(complete=True,plan=sha(R/'PLAN.json'),selected_variant=variant,selected_memory_arm=arm,
               confirmation_quality=quality,q0_confirmation_quality=q0quality,
               readout_replicated=bool(reader_replicated),q0_readout_replicated=bool(initial_replicated),
               residual_advantage_gate=bool(content_gate),recommendation=direction,reason=reason,
               discovery=earlier,confirmation=report,
               limitations=plan['limitations'],closed_loop_run=False)
    write(R/'FINAL_SUMMARY.json',value)
    print('NEXT_DIRECTION',direction,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['discovery','final']);a=p.parse_args();decision() if a.stage=='discovery' else final()
