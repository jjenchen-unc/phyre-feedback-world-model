"""Compare exact inference against frozen real checkpoints and saved policies."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch

PROJECT = Path('/path/to/phyre-unc')
MEM = PROJECT/'results/small_official_20261010'
MATCHED = PROJECT/'results/matched_base_q0_20261012'
sys.path.insert(0, str(MEM/'code'))
sys.path.insert(0, str(MEM/'recovery_20261010'))
import recovery_runner as m
from fast_inference import CandidateRanker, PreparedProbeReadout


def timing(fn):
    torch.cuda.synchronize(); start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter()-start


def code_hashes():
    return {name: hashlib.sha256((PROJECT/'pipeline'/name).read_bytes()).hexdigest()
            for name in ['fast_inference.py','benchmark_fast_inference.py']}


@torch.no_grad()
def run(output, shared, depth):
    source = runpy.run_path(str(MATCHED/'code/matched_base_q0.py'))
    plan = source['checked']()
    result = {'complete': False, 'shared_gpu_timing': shared, 'code': code_hashes(),
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'static': [], 'memory': []}
    for panel in ['cross','within']:
        tasks = plan['panels'][panel]['tasks']
        task = tasks[0]
        d, status, path = m.input_case(panel, task)
        z0 = torch.from_numpy(d['z0']).cuda().float()
        actions = torch.from_numpy(d['action']).cuda()
        features = [v[None] for v in d['features']]
        for arch in ['base','q0']:
            name = source['spec_name'](arch,.1,0)
            done = m.read(MATCHED/'models'/panel/name/'complete.json')
            assert m.sha(done['checkpoint']) == done['checkpoint_sha256']
            model, probe, head, rule, _ = source['models'](plan,panel,arch,0)
            ck = torch.load(done['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(ck['state_dict']); model.eval().requires_grad_(False)
            ranker = CandidateRanker(model,head,rule,z0,actions,features)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred = model(z0[:32],actions[:32])
            layout = PreparedProbeReadout(features[:32],pred.shape[1],pred.device)
            component = {'legacy': [], 'exact': []}
            for _ in range(3):
                a, ta = timing(lambda:m.p.f.readout(head,pred,features[:32],rule)['log_score'])
                b, tb = timing(lambda:layout(head,pred,rule))
                np.testing.assert_array_equal(a,b)
                component['legacy'].append(ta); component['exact'].append(tb)
            score = m.p.score_function(head,rule)
            # First call prepares layouts; report cold and warm end-to-end time.
            baseline, old_seconds = timing(lambda:m.p.static_scores(model,z0,actions,features,score))
            torch.cuda.reset_peak_memory_stats()
            fast, cold = timing(ranker.rank)
            again, warm = timing(ranker.rank)
            with np.load(MATCHED/'evaluation'/panel/name/task.replace(':','_')/'scores.npz') as old:
                np.testing.assert_array_equal(baseline,old['scores'])
            np.testing.assert_array_equal(baseline,fast.scores)
            np.testing.assert_array_equal(baseline,again.scores)
            assert fast.selected == int(np.argmax(baseline))
            row = {'panel':panel,'arch':arch,'task':task,'scores_exact':True,
                   'ranking_exact':True,'legacy_seconds':old_seconds,
                   'exact_cold_seconds':cold,'exact_warm_seconds':warm,
                   'warm_speedup':old_seconds/warm,'component_seconds':component,
                   'peak_GiB':torch.cuda.max_memory_allocated()/2**30,
                   'checkpoint_sha256':done['checkpoint_sha256']}
            result['static'].append(row)
            m.write(output,result); print('STATIC_PASS',row,flush=True)
            del model,probe,head,ranker,layout,ck,pred,fast,again
            gc.collect();torch.cuda.empty_cache()
        # Follow actual current-token updates, not a forced reference action list.
        folder = MEM/'evaluation_fixed'/panel/'residual_s0'
        task = next(t for t in tasks if m.read(folder/t.replace(':','_')/'result.json')['attempts']>=depth)
        d,status,path=m.input_case(panel,task)
        z0=torch.from_numpy(d['z0']).cuda().float();actions=torch.from_numpy(d['action']).cuda()
        features=[v[None] for v in d['features']]
        ref,model,head,rule,_=m.model_for(panel,'residual',0)
        model.eval().requires_grad_(False)
        ranker=CandidateRanker(model,head,rule,z0,actions,features)
        state=model.initial_state().clone();tried=np.zeros(1024,bool)
        with np.load(folder/task.replace(':','_')/'scores.npz') as z:
            expected=z['selected'].copy();reference=z['scores'].copy()
        selected=[]
        for i in range(min(depth,len(expected))):
            actual=ranker.rank(state,tried)
            np.testing.assert_array_equal(actual.scores,reference[i])
            assert actual.selected==int(expected[i])
            chosen=actual.selected;tried[chosen]=True;selected.append(chosen)
            if status[chosen]>0 or i+1==depth:break
            obs=MEM/'assets_fixed'/panel/'observations'/(task.replace(':','_')+f"_{int(d['action_idx'][chosen]):06d}.npz")
            with np.load(obs) as z:
                truth=torch.from_numpy(z['latents'][None].copy()).cuda().float()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                state=model.update_state(state,truth-actual.prediction,actions[chosen:chosen+1])
        row={'panel':panel,'task':task,'rounds':len(selected),'scores_exact':True,
             'selected_actions_exact':True,'selected':selected}
        result['memory'].append(row);m.write(output,result);print('MEMORY_PASS',row,flush=True)
        del ranker,model,ref,head
        gc.collect();torch.cuda.empty_cache()
    result['complete']=True;result['all_scores_and_actions_exact']=True
    m.write(output,result)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--shared-allocation',action='store_true')
    ap.add_argument('--depth',type=int,default=3)
    a=ap.parse_args();torch.set_num_threads(2)
    if a.depth<2 or a.depth>20:ap.error('depth must be between 2 and 20')
    run(a.output,a.shared_allocation,a.depth)
