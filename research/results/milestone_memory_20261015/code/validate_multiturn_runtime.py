"""Real-checkpoint integration checks and component timing, not method evaluation.

Run on one allocated GPU. Never write to previous model/evaluation directories.
Stress cases explicitly use failed-only candidates and are not success-rate data.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch

import matched_base_q0 as b
# matched_base_q0 prepends frozen dependency paths; keep NEW runtime explicit.
from multiturn_runtime import loss_all_updates, PredictorAdapter
from fast_inference import CandidateRanker, PreparedScorer
from updater_validation_numerics import controlled_backward

m, p = b.m, b.p
ROOT = Path('/path/to/phyre-unc')


def fingerprint(items):
    h = hashlib.sha256()
    for n, v in items:
        h.update(n.encode()); h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def timed(fn):
    torch.cuda.synchronize(); start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


class Telemetry:
    def __init__(self):
        self.samples = []; self.stop_event = threading.Event()
        self.uuid = str(torch.cuda.get_device_properties(0).uuid)
        self.thread = threading.Thread(target=self.loop, daemon=True)
    def loop(self):
        while not self.stop_event.is_set():
            sample = {'time': time.time(), 'own_pid': os.getpid(), 'allocated_uuid': self.uuid}
            for key, cmd in [
                ('gpus', ['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader']),
                ('processes', ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'])]:
                try: sample[key] = subprocess.check_output(cmd, text=True, timeout=8).strip()
                except Exception as e: sample[key] = repr(e)
            self.samples.append(sample)
            self.stop_event.wait(10)
    def __enter__(self): self.thread.start(); return self
    def __exit__(self, *args): self.stop_event.set(); self.thread.join(timeout=20)


def load(panel, arch='q0'):
    plan = m.read(b.ROOT/'PLAN.json')
    weight = 0. if panel == 'cross' else .03
    model, probe, head, rule, _ = b.models(plan, panel, arch, 0)
    info = m.read(b.ROOT/'models'/panel/b.spec_name(arch,weight,0)/'complete.json')
    assert m.sha(info['checkpoint']) == info['checkpoint_sha256']
    model.load_state_dict(torch.load(info['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
    model.eval().requires_grad_(False)
    if arch == 'q0': model.checkpoint_blocks = False
    return model, probe, head, rule, info


def dataset(panel):
    if panel == 'within': m.configure_within()
    manifest = m.read((m.W if panel=='within' else m.OLD)/'manifest.json')
    return p.f.Store(), [r for r in manifest['rows'] if r['split']=='train']


def episodes(rows, seed, count):
    result = []
    for i in range(count):
        row, qi, times, _ = p.episode(rows, seed, i)
        cap = int(np.random.default_rng([20261014,seed,i]).choice([1,2,4,8,19]))
        result.append((row, qi, times, cap))
    return result


def stress_episodes(source, rows):
    result = []
    for row in sorted(rows,key=lambda x:x['task']):
        failed = [i for i in row['candidates'] if not source.labels[i]]
        if len(failed) < 32: continue
        _, qi, times, _ = p.episode([row], 0, 0)
        stress = dict(row, candidates=failed[:32])
        result.append((stress, qi, times, 19))
        if len(result)==2: return result
    raise RuntimeError('Need two real failed-only 32-candidate stress scenes')


def optimizer_step(model, probe, head, rule, source, eps, opt, scheduler, fast, verify):
    opt.zero_grad(set_to_none=True); events = []; active = False; collect = 0.
    for ep in eps:
        loss, event, phase = loss_all_updates(model,probe,head,rule,source,ep,
                                             p.score_function(head,rule),fast=fast,verify=verify)
        assert torch.isfinite(loss)
        if loss.requires_grad: (loss/len(eps)).backward(); active = True
        events.append(event); collect += phase['collection_s']
        del loss
    if active:
        norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.)
        assert torch.isfinite(norm); opt.step()
    if scheduler is not None: scheduler.step()
    return {'events':events,'collection_s':collect,'optimizer_step':active}


def training(args, result, save):
    model,probe,head,rule,info = load(args.panel)
    source, rows = dataset(args.panel)
    model.set_stage('updater'); model.eval(); model.checkpoint_blocks = False
    initial = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    frozen = fingerprint((n,v) for n,v in model.named_parameters() if not v.requires_grad)
    probe_hash = fingerprint(probe.state_dict().items())
    natural = episodes(rows,0,2*args.verify_slots)
    stress = stress_episodes(source, rows)
    result.update(checkpoint_sha256=info['checkpoint_sha256'],training_equivalence={},training_timing={})
    result['candidate_counts'] = [len(r['candidates']) for r in rows]
    for zero in [False,True]:
        key = 'zero' if zero else 'residual'; result['training_equivalence'][key] = []
        for fast in [False,True]:
            model.load_state_dict(initial); model.zero_residual=zero
            torch.manual_seed(1901);torch.cuda.manual_seed_all(1901)
            opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4,weight_decay=.05)
            sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-4,total_steps=10000,pct_start=.03)
            arm={'backend':'fast' if fast else 'reference','slots':[]}
            result['training_equivalence'][key].append(arm)
            with controlled_backward('cuda',True):
                for i in range(args.verify_slots+1):
                    ep = natural[2*i:2*i+2] if i<args.verify_slots else stress
                    value,seconds=timed(lambda:optimizer_step(model,probe,head,rule,source,ep,opt,sched,fast,True))
                    if i==args.verify_slots: assert all(e['used_updates']==19 for e in value['events'])
                    value.update(seconds=seconds,stress=i==args.verify_slots,
                        gradient_sha256=fingerprint((n,v.grad) for n,v in model.named_parameters() if v.grad is not None),
                        weight_sha256=fingerprint((n,v) for n,v in model.named_parameters() if v.requires_grad))
                    arm['slots'].append(value);save()
                    print('VERIFY',args.panel,key,fast,i,[e['used_updates'] for e in value['events']],seconds,flush=True)
            assert frozen==fingerprint((n,v) for n,v in model.named_parameters() if not v.requires_grad)
            assert probe_hash==fingerprint(probe.state_dict().items())
            del opt,sched;gc.collect();torch.cuda.empty_cache()
        a,c=result['training_equivalence'][key]
        for x,y in zip(a['slots'],c['slots']):
            assert x['events']==y['events'],'Training action/support/loss mismatch'
            assert x['gradient_sha256']==y['gradient_sha256'],'Training gradient mismatch'
            assert x['weight_sha256']==y['weight_sha256'],'Training weights mismatch'
        result.setdefault('checks',{})[key+'_train_exact']=True;save()

        # Actual no-audit training trajectory. Timing includes loading, support
        # ranking, complete replay, every-state query loss, backward and AdamW.
        model.load_state_dict(initial); model.zero_residual=zero
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4,weight_decay=.05)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-4,total_steps=10000,pct_start=.03)
        draw=episodes(rows,3,2*args.bench_slots)
        result['training_timing'][key]=[]
        torch.cuda.reset_peak_memory_stats()
        with controlled_backward('cuda',True):
            for i in range(args.bench_slots):
                value,seconds=timed(lambda:optimizer_step(model,probe,head,rule,source,draw[2*i:2*i+2],opt,sched,True,False))
                value.update(seconds=seconds,warmup=i<4,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
                result['training_timing'][key].append(value);save()
                print('BENCH',args.panel,key,i,[e['used_updates'] for e in value['events']],seconds,flush=True)
        del opt,sched;gc.collect();torch.cuda.empty_cache()
    result['complete']=True;save()


def selected_prediction(model,z0,actions,chosen,state):
    start=chosen//32*32;end=min(start+32,len(actions))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        if state is None: pred=model(z0[start:end],actions[start:end])
        else: pred=model(z0[start:end],actions[start:end],state.expand(end-start,-1,-1))
    return pred[chosen-start:chosen-start+1]


def trajectory(model,head,rule,z0,actions,features,observe,kind,fast,steps=20,lr=3e-4,tta_steps=1,audit_turns=()):
    adapter=PredictorAdapter(model,lr=lr,steps=tta_steps) if kind=='tta' else None
    if adapter is None: model.requires_grad_(False)
    model.eval()
    state=model.initial_state().detach().clone() if kind in ['residual','zero'] else None
    ranker=CandidateRanker(model,head,rule,z0,actions,features)
    tried=np.zeros(len(actions),bool);records=[]
    for turn in range(steps):
        def rank():
            if fast:
                r=ranker.rank(state,tried);return r.scores,r.selected,r.prediction
            with torch.no_grad():
                if state is None:
                    scores=p.static_scores(model,z0,actions,features,p.score_function(head,rule))
                    masked=scores.copy();masked[tried]=-np.inf;j=int(masked.argmax())
                    return scores,j,selected_prediction(model,z0,actions,j,None)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    j,scores,pred=p.choose_action(model,state,z0,actions,features,p.score_function(head,rule),torch.from_numpy(tried).cuda())
                return scores.cpu().numpy(),j,pred
        (scores,j,pred),rank_s=timed(rank)
        assert j is not None and not tried[j];tried[j]=True
        record={'turn':turn,'chosen':j,'score_sha256':hashlib.sha256(scores.tobytes()).hexdigest(),'rank_s':rank_s}
        if turn in audit_turns:
            def original_scores():
                with torch.no_grad():
                    if state is None:
                        return p.static_scores(model,z0,actions,features,p.score_function(head,rule))
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        _, values, _ = p.choose_action(model,state,z0,actions,features,p.score_function(head,rule),torch.zeros(len(actions),device=z0.device,dtype=torch.bool))
                    return values.cpu().numpy()
            reference,audit_s=timed(original_scores)
            np.testing.assert_array_equal(scores,reference)
            record['audit_s']=audit_s
            record['full_pool_scores_exact']=True
        if turn<steps-1:
            truth,obs_s=timed(lambda:observe(j))
            if adapter:
                event,update_s=timed(lambda:adapter.update(j,z0[j:j+1],actions[j:j+1],truth))
                assert event['replay_indices']==[r['chosen'] for r in records[max(0,len(records)-4):]]+[j]
                record['adaptation']=event
            else:
                def update():
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        return model.update_state(state,truth-pred,actions[j:j+1])
                state,update_s=timed(update)
            record.update(observation_s=obs_s,update_s=update_s)
        if adapter:
            record['state_sha256']=fingerprint((n,v) for n,v in model.named_parameters() if v.requires_grad)
        else:record['state_sha256']=fingerprint([('tokens',state)])
        records.append(record)
    if adapter:adapter.reset()
    return records


def evaluation(args,result,save):
    model,probe,head,rule,info=load(args.panel)
    source,rows=dataset(args.panel)
    stress=stress_episodes(source,rows)
    row,qi,times,_=stress[0];ids=row['candidates']
    z0,actions,features=p.initial_data(source,ids)
    observe=lambda j:torch.from_numpy(source.z([ids[j]])).cuda().float()
    initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    result.update(checkpoint_sha256=info['checkpoint_sha256'],short_pool={},full_pool={})
    for kind in ['residual','zero','tta']:
        model.load_state_dict(initial); model.zero_residual=kind=='zero'
        if kind!='tta':
            model.set_stage('updater')
            opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4,weight_decay=.05)
            # Exercise a NON-identity updater on the selected NEW Q0 checkpoint.
            with controlled_backward('cuda',True):
                for i in range(3):
                    eps=[(r,q,t,2) for r,q,t,_ in stress]
                    optimizer_step(model,probe,head,rule,source,eps,opt,None,True,True)
            del opt
        start={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        result['short_pool'][kind]=[]
        for fast in [False,True]:
            model.load_state_dict(start)
            with controlled_backward('cuda',True):
                value,seconds=timed(lambda:trajectory(model,head,rule,z0,actions,features,observe,kind,fast))
            result['short_pool'][kind].append({'backend':'fast' if fast else 'reference','events':value,'seconds':seconds})
            save()
        x,y=[a['events'] for a in result['short_pool'][kind]]
        for a,c in zip(x,y):
            for key in ['chosen','score_sha256','state_sha256','adaptation']:
                assert a.get(key)==c.get(key),(kind,key,a,c)
        result.setdefault('checks',{})[kind+'_19update_exact']=True
        if kind=='tta':
            assert fingerprint(model.state_dict().items())==fingerprint(start.items()),'TTA reset failed'
        model.load_state_dict(start)
        # Save short-trained updaters ONLY as validation artifacts, not scientific models.
        b.atomic_save(args.output.parent/f'{args.panel}_{kind}_smoke.pt',{'state_dict':start,'validation_only':True})
        save()
    # Full real 1024 pool: use an explicitly labelled zero-success stress scene,
    # so twenty inference rounds are guaranteed without pretending efficacy.
    plan=m.read(b.ROOT/'PLAN.json');task=None
    for candidate in plan['panels'][args.panel]['tasks']:
        d,status,path=m.input_case(args.panel,candidate)
        if not (status>0).any():task=candidate;break
    assert task is not None and len(d['action'])==1024
    result['full_pool_task']=task;result['full_pool_role']='historical shortlist, all-failure integration stress, not method performance'
    z=torch.from_numpy(d['z0']).cuda().float();a=torch.from_numpy(d['action']).cuda();feats=[v[None] for v in d['features']]
    observations=m.Observations()
    observation_access=[]
    def obs(j):
        cache=m.ASSETS/args.panel/'observations'/f'{task.replace(":","_")}_{int(d["action_idx"][j]):06d}.npz'
        observation_access.append({'action_idx':int(d['action_idx'][j]),'cache_hit':cache.exists()})
        return observations.get(args.panel,task,int(d['action_idx'][j]),d['action'][j],d['z0'][j],int(status[j]))
    try:
        for kind in ['residual','zero','tta']:
            model.load_state_dict(torch.load(args.output.parent/f'{args.panel}_{kind}_smoke.pt',map_location='cpu',weights_only=False)['state_dict'])
            model.zero_residual=kind=='zero';model.eval().requires_grad_(False)
            snapshot={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            # Interleave reference/fast complete ranking timings, with warmup.
            rs=[]
            for fast in [False,True,True,False,False,True]:
                ranker=CandidateRanker(model,head,rule,z,a,feats)
                if fast:(value,seconds)=timed(lambda:ranker.rank().scores)
                else:(value,seconds)=timed(lambda:p.static_scores(model,z,a,feats,p.score_function(head,rule)))
                rs.append({'fast':fast,'seconds':seconds,'scores_sha256':hashlib.sha256(value.tobytes()).hexdigest()})
            assert len({x['scores_sha256'] for x in rs})==1
            result['full_pool'][kind]={'rank_abba':rs}
            observation_access.clear()
            torch.cuda.reset_peak_memory_stats()
            with controlled_backward('cuda',True):
                value,seconds=timed(lambda:trajectory(model,head,rule,z,a,feats,obs,kind,True,audit_turns=(0,1,19)))
            result['full_pool'][kind].update(events=value,seconds=seconds,observation_access=observation_access.copy(),
                seconds_excluding_score_audit=seconds-sum(e.get('audit_s',0) for e in value),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
            assert len(value)==20
            if kind=='tta':assert fingerprint(model.state_dict().items())==fingerprint(snapshot.items())
            save();print('FULL_POOL',args.panel,kind,seconds,flush=True)
        # Q0 TTA includes replay lengths1..5; time 1 and3 gradient steps explicitly.
        model.load_state_dict(initial)
        result['tta_micro']={}
        for steps in [1,3]:
            adapter=PredictorAdapter(model,steps=steps)
            records=[]
            with controlled_backward('cuda',True):
                for j in range(7):
                    truth=observe(j)
                    event,seconds=timed(lambda:adapter.update(j,z0[j:j+1],actions[j:j+1],truth))
                    records.append({'seconds':seconds,**event})
            adapter.reset();result['tta_micro'][str(steps)]=records;save()
    finally: observations.close()
    # Backwards compatibility: new wrapper-capable adapter must exactly match
    # the old adapter on the new bare-base checkpoint for replay1..5 and1/3steps.
    import runpy
    OldAdapter=runpy.run_path(str(ROOT/'results/adajepa_phyre_20261010/code/tta.py'))['Adapter']
    del model,probe,head;gc.collect();torch.cuda.empty_cache()
    base,probe,head,rule,base_info=load(args.panel,'base')
    initial_base={n:v.detach().cpu().clone() for n,v in base.state_dict().items()}
    result['base_adapter_compatibility']={}
    for steps in [1,3]:
        arms=[]
        for cls in [OldAdapter,PredictorAdapter]:
            base.load_state_dict(initial_base);adapter=cls(base,3e-4,steps)
            records=[]
            with controlled_backward('cuda',True):
                for j in range(7):
                    truth=observe(j)
                    event,seconds=timed(lambda:adapter.update(j,z0[j:j+1],actions[j:j+1],truth))
                    records.append({'losses':event['losses'],'replay_indices':event['replay_indices'],'seconds':seconds,
                        'weights':fingerprint(base.state_dict().items())})
            adapter.reset();assert fingerprint(base.state_dict().items())==fingerprint(initial_base.items())
            arms.append(records)
        for x,y in zip(*arms):
            for k in ['losses','replay_indices','weights']:assert x[k]==y[k]
        result['base_adapter_compatibility'][str(steps)]={'exact':True,'arms':arms};save()
    result['complete']=True;save()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',choices=['cross','within'],required=True)
    ap.add_argument('--mode',choices=['train','eval'],required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--verify-slots',type=int,default=12);ap.add_argument('--bench-slots',type=int,default=64)
    args=ap.parse_args();assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    result={'complete':False,'panel':args.panel,'mode':args.mode,'started':time.time(),'job':os.environ.get('SLURM_JOB_ID'),
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'args':vars(args)|{'output':str(args.output)},
            'code':{n:m.sha(ROOT/'pipeline'/n) for n in ['multiturn_runtime.py','validate_multiturn_runtime.py','fast_inference.py','updater_validation_numerics.py']}}
    telemetry=Telemetry()
    def save():
        result['telemetry']=telemetry.samples.copy();result['elapsed_s']=time.time()-result['started']
        m.write(args.output,result)
    try:
        with telemetry:
            if args.mode=='train':training(args,result,save)
            else:evaluation(args,result,save)
    except Exception as e:
        result['error']=repr(e);save();raise
    finally:save()


if __name__=='__main__':main()
