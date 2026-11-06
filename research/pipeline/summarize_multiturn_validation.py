"""Re-audit raw comparisons and derive explicitly conditional cost projections."""
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/multiturn_validation_20261014'


def read(name):
    return json.loads((RUN/(name+'.json')).read_text())


def stats(values):
    values = sorted(values)
    return dict(n=len(values), mean=st.mean(values), median=st.median(values),
                p90=values[min(len(values)-1, int(.9*len(values)))],
                minimum=values[0], maximum=values[-1])


def telemetry(samples):
    shared = []
    for item in samples:
        uuid = 'GPU-'+item['allocated_uuid'].removeprefix('GPU-')
        others = [line for line in item['processes'].splitlines()
                  if line.split(',')[0].strip() == uuid
                  and int(line.split(',')[1]) != item['own_pid']]
        if others: shared.append({'time':item['time'], 'other_processes':others})
    return {'samples':len(samples), 'samples_with_other_GPU_processes':len(shared),
            'shared':shared}


def main():
    summary = {'audit_complete':False, 'training':{}, 'evaluation':{}, 'initial':{},
               'telemetry':{}, 'inputs_sha256':{}, 'cost_scenarios':{}}
    for panel in ['cross','within']:
        train = read('train_'+panel)
        assert train['complete'] and not train.get('error')
        summary['training'][panel] = {}
        for kind, arms in train['training_equivalence'].items():
            assert len(arms)==2
            reference, fast = [a['slots'] for a in arms]
            assert len(reference)==len(fast)==13
            for x,y in zip(reference,fast):
                for key in ['events','gradient_sha256','weight_sha256','optimizer_step']:
                    assert x[key]==y[key], (panel,kind,key)
                for event in x['events']:
                    assert not set(event['queries']) & set(event['selected'])
                    assert len(event['selected'])==len(set(event['selected']))
                    assert len(event['state_terms'])==max(1,event['used_updates'])
            assert all(e['used_updates']==19 for e in fast[-1]['events'])
            rows=[r for r in train['training_timing'][kind] if not r['warmup']]
            assert len(rows)==60
            events=[e for r in rows for e in r['events']]
            times=stats([r['seconds'] for r in rows])
            summary['training'][panel][kind] = {
                'exact_slots':len(fast), 'exact_episodes':2*len(fast),
                'slot_seconds':times, 'updates_per_episode':stats([e['used_updates'] for e in events]),
                'zero_update_episodes':sum(e['used_updates']==0 for e in events),
                'zero_gradient_slots':sum(not r['optimizer_step'] for r in rows),
                'cap_counts':{str(k):sum(e['cap']==k for e in events) for k in [1,2,4,8,19]},
                'peak_natural_GiB':max(r['peak_GiB'] for r in rows),
                'two_19update_stress_seconds':fast[-1]['seconds'],
                '10000_slots_h_at_sample_distribution':times['mean']*10000/3600,
            }
        ev=read('eval_v2_'+panel)
        assert ev['complete'] and not ev.get('error')
        summary['evaluation'][panel]={}
        assert len({ev['full_pool'][k]['rank_abba'][0]['scores_sha256']
                    for k in ['residual','zero','tta']})==1
        summary['evaluation'][panel]['shared_Q0_initial_scores_exact']=True
        for kind in ['residual','zero','tta']:
            x,y=[a['events'] for a in ev['short_pool'][kind]]
            assert len(x)==len(y)==20
            for a,b in zip(x,y):
                for key in ['chosen','score_sha256','state_sha256','adaptation']:
                    assert a.get(key)==b.get(key)
            full=ev['full_pool'][kind]
            events=full['events']
            assert len(events)==20 and len(set(e['chosen'] for e in events))==20
            assert all(events[i]['full_pool_scores_exact'] for i in [0,1,19])
            ranks=full['rank_abba']
            assert len({r['scores_sha256'] for r in ranks})==1
            ref=st.mean(r['seconds'] for r in ranks[2:] if not r['fast'])
            fast=st.mean(r['seconds'] for r in ranks[2:] if r['fast'])
            warmed_audits=[events[i] for i in [1,19]]
            controlled_ref=st.mean(e['audit_s'] for e in warmed_audits)
            controlled_fast=st.mean(e['rank_s'] for e in warmed_audits)
            summary['evaluation'][panel][kind]={
                'short_pool_exact_turns':20, 'full_pool_exact_score_turns':[0,1,19],
                'static_reference_rank_s':ref, 'static_fast_rank_s':fast,
                'static_rank_speedup_ratio':ref/fast,
                'static_rank_time_reduction_fraction':1-fast/ref,
                'controlled_warmed_reference_rank_s':controlled_ref,
                'controlled_warmed_fast_rank_s':controlled_fast,
                'controlled_warmed_rank_time_reduction_fraction':1-controlled_fast/controlled_ref,
                'closed_loop_rank_s':stats([e['rank_s'] for e in events]),
                'closed_loop_update_s':stats([e['update_s'] for e in events[:-1]]),
                'observation_s':stats([e['observation_s'] for e in events[:-1]]),
                'observation_cache_hits':sum(e['cache_hit'] for e in full['observation_access']),
                'seconds_20trials_excluding_score_audit':full['seconds_excluding_score_audit'],
                'peak_GiB':full['peak_GiB'],
            }
        summary['evaluation'][panel]['tta_step_micro']={
            k:stats([r['seconds'] for r in rows[4:]]) for k,rows in ev['tta_micro'].items()}
        for steps, data in ev['base_adapter_compatibility'].items():
            assert data['exact']
            for a,b in zip(*data['arms']):
                for key in ['losses','replay_indices','weights']:assert a[key]==b[key]
        summary['evaluation'][panel]['base_adapter_exact_steps']=[1,3]
        depth=read('tta_depth_'+panel)
        assert depth['complete'] and depth['lr']==1e-3 and depth['steps']==3
        for arch,arms in depth['arms'].items():
            assert all(a['reset_exact'] for a in arms)
            x,y=[a['events'] for a in arms]
            assert len(x)==len(y)==20
            for a,b in zip(x,y):
                for key in ['chosen','score_sha256','state_sha256','adaptation']:
                    assert a.get(key)==b.get(key)
        summary['evaluation'][panel]['maximum_planned_TTA_depth_exact']={
            'architectures':['base','q0'],'lr':1e-3,'steps':3,'updates':19}
        initial=read('initial_'+panel)
        assert initial['complete']
        summary['initial'][panel]={}
        for arch, arm in initial['arms'].items():
            step=stats([r['seconds'] for r in arm['steps'] if not r['warmup']])
            steps=52272 if panel=='cross' else 52344
            estimated=(step['mean']*steps + arm['dev400_s']*9 + arm['checkpoint_write_s']*61 + initial['pack_load_s'])/3600
            summary['initial'][panel][arch]={
                'step_seconds':step, 'dev400_s':arm['dev400_s'],
                'checkpoint_write_s':arm['checkpoint_write_s'], 'pack_load_s':initial['pack_load_s'],
                'peak_GiB':arm['peak_GiB'], '8epoch_h_at_sample_throughput':estimated}
    for p in sorted(RUN.glob('*.json')):
        if p.name=='SUMMARY.json':continue
        summary['inputs_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        d=json.loads(p.read_text())
        if 'telemetry' in d:summary['telemetry'][p.name]=telemetry(d['telemetry'])
    data=read('data_cost');reuse=read('data_reuse');repeat=read('repeated_simulation')
    assert data['complete'] and reuse['complete'] and repeat['complete']
    assert read('reuse_latent_check')['complete']
    rows=[r for r in data['samples'] if not r['warmup']]
    encode=stats([r['trajectory_encode_s'] for r in rows])
    initial_encode=stats([r['initial_encode_s'] for r in rows])
    summary['data']={
        'trajectory_encode_s':encode, 'one_initial_encode_s':initial_encode,
        'warm_same_scene_sim_save_s':stats([r['seconds'] for r in repeat['rows'] if r['repeat']>0]),
        'new_training_encode_h_if_reuse_validated':reuse['union']['files_needing_encoding']*encode['mean']/3600,
        '600x1024_initial_encode_h_no_cross_panel_reuse':600*1024*initial_encode['mean']/3600,
        'new_unique_trajectory_latent_GiB':reuse['union']['files_needing_encoding']*rows[0]['latent_bytes']/2**30,
        'reuse_inventory':{k:v for k,v in reuse['union'].items() if k!='missing_keys'}}
    initial_h=2*sum(v['8epoch_h_at_sample_throughput'] for p in summary['initial'].values() for v in p.values())
    updater_h=3*sum(v['10000_slots_h_at_sample_distribution'] for p in summary['training'].values() for v in p.values())
    # Scenarios are allocations, not confidence intervals or success predictions.
    # Use Q0 TTA as an explicit conservative proxy for unmeasured bare-base full-pool rank.
    for trials in [5,10,20]:
        total=calibration=diagnostics=0.
        for panel in ['cross','within']:
            ev=summary['evaluation'][panel]
            static=600*ev['residual']['static_fast_rank_s']
            dynamic=0.
            for kind,number in [('residual',300),('zero',300),('tta',600)]:
                d=ev[kind]
                update=d['closed_loop_update_s']['mean'] if kind!='tta' else ev['tta_step_micro']['3']['mean']
                # Cold observation proxy uses measured simulation+encoding; cache helps.
                obs=summary['data']['warm_same_scene_sim_save_s']['mean']+encode['mean']
                dynamic+=number*(trials*d['closed_loop_rank_s']['mean']+(trials-1)*(obs+update))
            total+=(static+dynamic)/3600
            tta=ev['tta'];r=tta['closed_loop_rank_s']['mean']
            mean_tta_update=st.mean(ev['tta_step_micro'][str(k)]['mean'] for k in [1,3])
            calibration+=300*(trials*r+(trials-1)*(obs+mean_tta_update))/3600
            r=ev['residual']['closed_loop_rank_s']['mean']
            update=ev['residual']['closed_loop_update_s']['mean']
            diagnostics+=300*(trials*r+(trials-1)*(obs+update)+2*r+obs+update)/3600
        preparation=summary['data']['new_training_encode_h_if_reuse_validated']+summary['data']['600x1024_initial_encode_h_no_cross_panel_reuse']
        summary['cost_scenarios'][str(trials)]={'assumed_mean_dynamic_trials':trials,
            'new_initial_trainings_GPU_h':initial_h,'updater_GPU_h_if_early_success_distribution_persists':updater_h,
            'main_36_evaluations_GPU_allocated_h':total,
            '600_TTA_calibration_episodes_GPU_allocated_h':calibration,
            '1200_online_diagnostic_episodes_GPU_allocated_h':diagnostics,
            'new_training_and_main_pool_encoding_GPU_h':preparation,
            'subtotal_excludes_new_heads_offline_diagnostics_queue_h':initial_h+updater_h+total+calibration+diagnostics+preparation}
    summary['historical_training_GPU_h']=sum(r['train_h'] for r in read('HISTORICAL_TIMING')['rows'])
    summary['summary_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['audit_complete']=True
    (RUN/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ['audit_complete','cost_scenarios','data']},indent=2))


if __name__=='__main__':main()
