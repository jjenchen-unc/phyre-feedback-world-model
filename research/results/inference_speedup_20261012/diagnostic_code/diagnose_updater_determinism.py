"""Paired real updater verification; persist diagnostics before equality gates."""
import argparse
import gc
import hashlib
from pathlib import Path
import sys
import time

import torch

PROJECT = Path(__file__).resolve().parents[1]
MEM = PROJECT/'results/small_official_20261010'
sys.path.insert(0, str(MEM/'code'))
sys.path.insert(0, str(MEM/'recovery_20261010'))
import recovery_runner as m
from fast_updater import updater_loss

p = m.p


def run(panel, output, slots, shared, reference_only=False, deterministic=False, variant="both"):
    import fast_updater
    if variant=="cache_only": fast_updater.PreparedScorer=p.score_function
    optimized_loss=lambda *a,**kw:updater_loss(*a,cache_observations=variant!="score_only",**kw)
    if deterministic: torch.use_deterministic_algorithms(True, warn_only=True)
    if panel == 'within':
        m.configure_within()
    manifest = m.read((m.W if panel == 'within' else m.OLD)/'manifest.json')
    rows = [r for r in manifest['rows'] if r['split']=='train']
    source = p.f.Store()
    episodes = [p.episode(rows,0,i) for i in range(2*slots)]
    ref, model, head, rule, _ = m.model_for(panel,'residual',0)
    model.train(); model.checkpoint_blocks = True
    ck = torch.load(m.PROBES[panel],map_location='cpu',weights_only=False)
    probe = m.AttnProbe(ck['D'],ck['S']).cuda()
    probe.load_state_dict(ck['state_dict']); probe.eval().requires_grad_(False)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    probe_initial = {k:v.detach().cpu().clone() for k,v in probe.state_dict().items()}
    trainable = {k for k,v in model.named_parameters() if v.requires_grad}
    # Warm immutable source pages for both arms; no updates or ranking warmup.
    for row, qi, _, _ in episodes:
        for idx in set(row['candidates']+qi):
            source.z([idx]); source.feature(idx)
    result = {'complete':False,'panel':panel,'slots':slots,'shared_gpu_timing':shared,'reference_only':reference_only,'deterministic_warn_only':deterministic,'variant':variant,
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
              'code':{name:hashlib.sha256((PROJECT/'pipeline'/name).read_bytes()).hexdigest()
                      for name in ['fast_inference.py','fast_updater.py','benchmark_fast_updater.py','diagnose_updater_determinism.py']},
              'arms':[]}
    grads, weights = [], []
    for name, loss_fn in [('reference',p.updater_loss),('reference_repeat' if reference_only else 'optimized',p.updater_loss if reference_only else optimized_loss)]:
        model.load_state_dict(initial)
        opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4,weight_decay=.05)
        arm = {'name':name,'events':[],'seconds':[],'optimizer_steps':0}
        arm_grads=[]
        torch.cuda.reset_peak_memory_stats()
        for slot in range(slots):
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); start=time.perf_counter()
            events=[]; active=False
            for ep in episodes[2*slot:2*slot+2]:
                loss,event=loss_fn(model,probe,head,rule,source,ep,verify=True)
                events.append(event)
                if loss.requires_grad:
                    (loss/2).backward(); active=True
            if active:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step(); arm['optimizer_steps']+=1
            torch.cuda.synchronize()
            arm['seconds'].append(time.perf_counter()-start)
            arm['events'].append(events)
            arm_grads.append({k:v.grad.detach().cpu().clone() for k,v in model.named_parameters()
                              if v.requires_grad and v.grad is not None})
            result['in_progress']={'arm':name,'completed_slots':slot+1,'events':events}
            m.write(output,result)
            print('SLOT',panel,name,slot,[e['used_updates'] for e in events],arm['seconds'][-1],flush=True)
        arm['peak_GiB']=torch.cuda.max_memory_allocated()/2**30
        arm['frozen_predictor_exact']=all(torch.equal(v.detach().cpu(),initial[k])
                                          for k,v in model.named_parameters() if k not in trainable)
        arm['probe_exact']=all(torch.equal(v.detach().cpu(),probe_initial[k]) for k,v in probe.state_dict().items())
        result['arms'].append(arm); grads.append(arm_grads)
        weights.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        m.write(output,result)
        del opt; gc.collect(); torch.cuda.empty_cache()
    result['events_exact']=result['arms'][0]['events']==result['arms'][1]['events']
    result['gradient_mismatches']=[{'slot':i,'parameter':k,'max_abs':float((a[k]-b[k]).abs().max()) if k in a and k in b else None} for i,(a,b) in enumerate(zip(*grads))
                                  for k in a.keys()|b.keys() if k not in a or k not in b or not torch.equal(a[k],b[k])]
    result['weight_mismatches']=[k for k in weights[0] if not torch.equal(weights[0][k],weights[1][k])]
    result['gradients_exact']=not result['gradient_mismatches']
    result['weights_exact']=not result['weight_mismatches']
    result['complete']=all(result[k] for k in ['events_exact','gradients_exact','weights_exact']) and all(
        a['probe_exact'] and a['frozen_predictor_exact'] for a in result['arms'])
    result['measured_speedup']=sum(result['arms'][0]['seconds'])/sum(result['arms'][1]['seconds'])
    result.pop('in_progress',None); m.write(output,result)
    assert result['complete'], f'Equivalence failed; diagnostics in {output}'
    print('PASS',panel,'events/loss/gradients/weights exact',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',choices=['cross','within'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--shared-allocation',action='store_true')
    parser.add_argument('--reference-only',action='store_true')
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--variant',choices=['both','cache_only','score_only'],default='both')
    args=parser.parse_args()
    if args.slots<1: parser.error('--slots must be positive')
    torch.set_num_threads(2)
    run(args.panel,args.output,args.slots,args.shared_allocation,args.reference_only,args.deterministic,args.variant)
