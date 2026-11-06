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


def run(panel, output, slots, shared):
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
    result = {'complete':False,'panel':panel,'slots':slots,'shared_gpu_timing':shared,
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
              'code':{name:hashlib.sha256((PROJECT/'pipeline'/name).read_bytes()).hexdigest()
                      for name in ['fast_inference.py','fast_updater.py','benchmark_fast_updater.py']},
              'arms':[]}
    grads, weights = [], []
    for name, loss_fn in [('reference',p.updater_loss),('optimized',updater_loss)]:
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
    result['gradient_mismatches']=[{'slot':i,'parameter':k} for i,(a,b) in enumerate(zip(*grads))
                                  for k in a.keys()|b.keys() if k not in a or k not in b or not torch.equal(a[k],b[k])]
    result['weight_mismatches']=[k for k in weights[0] if not torch.equal(weights[0][k],weights[1][k])]
    result['gradients_exact']=not result['gradient_mismatches']
    result['weights_exact']=not result['weight_mismatches']
    # Original-versus-original CUDA reruns also vary at ~1e-10. Keep the exact
    # fields, and separately gate numerical equivalence at explicit FP32 bounds.
    def delta(a,b):
        return {'max_abs':float((a-b).abs().max()),
                'close':bool(torch.allclose(a,b,rtol=1e-6,atol=1e-8))}
    result['tolerance']={'rtol':1e-6,'atol':1e-8}
    result['gradient_deltas']=[{'slot':i,'parameter':k,**delta(a[k],b[k])}
                               for i,(a,b) in enumerate(zip(*grads)) for k in a.keys()&b.keys()
                               if not torch.equal(a[k],b[k])]
    result['weight_deltas']=[{'parameter':k,**delta(weights[0][k],weights[1][k])}
                            for k in result['weight_mismatches']]
    result['gradients_close']=(all(a.keys()==b.keys() for a,b in zip(*grads)) and
                               all(d['close'] for d in result['gradient_deltas']))
    result['weights_close']=all(d['close'] for d in result['weight_deltas'])
    result['complete']=all(result[k] for k in ['events_exact','gradients_close','weights_close']) and all(
        a['probe_exact'] and a['frozen_predictor_exact'] for a in result['arms'])
    result['measured_speedup']=sum(result['arms'][0]['seconds'])/sum(result['arms'][1]['seconds'])
    result.pop('in_progress',None); m.write(output,result)
    assert result['complete'], f'Equivalence failed; diagnostics in {output}'
    print('PASS',panel,'events/loss exact; gradients/weights within recorded tolerance',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',choices=['cross','within'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--shared-allocation',action='store_true')
    args=parser.parse_args()
    if args.slots<1: parser.error('--slots must be positive')
    torch.set_num_threads(2)
    run(args.panel,args.output,args.slots,args.shared_allocation)
