"""Accelerated static evaluation of completed matched-study base/Q0 checkpoints.

Writes a separate, provenance-tagged evaluation; never changes a registered run.
Uses the unchanged fixed candidate pool, labels, FP32 scorer, and B32 predictor.
"""
import argparse
import hashlib
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch

PROJECT=Path('/path/to/phyre-unc')
MEM=PROJECT/'results/small_official_20261010'
MATCHED=PROJECT/'results/matched_base_q0_20261012'
sys.path.insert(0,str(MEM/'code'))
sys.path.insert(0,str(MEM/'recovery_20261010'))
import recovery_runner as m
from attribution_a_metrics import auc
from fast_inference import CandidateRanker


@torch.no_grad()
def evaluate(panel,arch,weight,seed,output,verify_reference,limit_tasks=None):
    source=runpy.run_path(str(MATCHED/'code/matched_base_q0.py'))
    plan=source['checked']();name=source['spec_name'](arch,weight,seed)
    if any(output.resolve().is_relative_to(root.resolve()) for root in [MATCHED,MEM]):
        raise ValueError('Use a separate output directory; registered evaluation is immutable')
    tasks=plan['panels'][panel]['tasks'][:limit_tasks]
    done=m.read(MATCHED/'models'/panel/name/'complete.json')
    assert done['complete'] and m.sha(done['checkpoint'])==done['checkpoint_sha256']
    model,probe,head,rule,_=source['models'](plan,panel,arch,seed)
    ck=torch.load(done['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state_dict']);model.eval().requires_grad_(False)
    provenance={'training':done['provenance'],'checkpoint_sha256':done['checkpoint_sha256'],
                'backend':'prepared_probe_exact','batch':32,'tasks':tasks,
                'code':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__),Path(__file__).with_name('fast_inference.py')]}}
    output.mkdir(parents=True,exist_ok=True);rows=[];t0=time.perf_counter()
    for task in tasks:
        d,status,pool=m.input_case(panel,task);path=output/task.replace(':','_')
        prov={**provenance,'shortlist_sha256':m.sha(pool),
              'labels_sha256':m.sha(pool.with_name(pool.stem+'_labels.npy'))}
        if (path/'result.json').exists():
            row=m.read(path/'result.json');assert row['provenance']==prov
        else:
            z0=torch.from_numpy(d['z0']).cuda().float();actions=torch.from_numpy(d['action']).cuda()
            features=[v[None] for v in d['features']]
            ranker=CandidateRanker(model,head,rule,z0,actions,features)
            torch.cuda.synchronize();start=time.perf_counter();ranked=ranker.rank()
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            path.mkdir(parents=True,exist_ok=True)
            m.p.f.save_npz(path/'scores.npz',{'indices':d['action_idx'],'labels':status>0,'scores':ranked.scores})
            row={'task':task,'provenance':prov,'ranking_seconds':elapsed,
                 'first_success':m.p.first_rank(ranked.scores,status>0),'auc':auc(status>0,ranked.scores)}
            m.write(path/'result.json',row)
        with np.load(path/'scores.npz') as z:
            np.testing.assert_array_equal(z['indices'],d['action_idx'])
            np.testing.assert_array_equal(z['labels'],status>0)
            assert m.p.first_rank(z['scores'],z['labels'])==row['first_success']
            assert auc(z['labels'],z['scores'])==row['auc']
            if verify_reference:
                ref=MATCHED/'evaluation'/panel/name/task.replace(':','_')/'scores.npz'
                with np.load(ref) as r:
                    for key in ['indices','labels','scores']:np.testing.assert_array_equal(z[key],r[key])
        rows.append(row)
        m.write(output/'progress.json',{'scenes':len(rows),'total':len(tasks)})
        print('FAST_EVAL',panel,name,len(rows),task,row['first_success'],flush=True)
    counts={f's{k}':sum(r['first_success'] is not None and r['first_success']<=k for r in rows)
            for k in [1,3,5,9,20]}
    aucs=[r['auc'] for r in rows if r['auc'] is not None]
    m.write(output/'complete.json',{'complete':True,'raw_audit_passed':True,
             'scope':'full_panel' if len(tasks)==len(plan['panels'][panel]['tasks']) else 'subset',
             'reference_scores_verified':verify_reference,'n':len(rows),'counts':counts,
             'mean_auc':float(np.mean(aucs)) if aucs else None,
             'auc_n':sum(r['auc'] is not None for r in rows),'provenance':provenance,
             'ranking_seconds':sum(r['ranking_seconds'] for r in rows),
             'seconds_this_invocation':time.perf_counter()-t0})


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--panel',choices=['cross','within'],required=True)
    ap.add_argument('--arch',choices=['base','q0'],required=True)
    ap.add_argument('--weight',type=float,default=.1)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--verify-reference',action='store_true')
    ap.add_argument('--limit-tasks',type=int,help='Smoke subset; recorded in provenance and scope')
    a=ap.parse_args();torch.set_num_threads(4)
    if a.limit_tasks is not None and a.limit_tasks<1:ap.error('--limit-tasks must be positive')
    evaluate(a.panel,a.arch,a.weight,a.seed,a.output,a.verify_reference,a.limit_tasks)
