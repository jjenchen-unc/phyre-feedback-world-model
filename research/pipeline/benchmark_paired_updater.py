"""Compare each updater step from identical weights AND optimizer state.

Free-running arms can amplify non-bit-exact CUDA gradients into policy divergence.
This test isolates implementation effects by restoring the reference input state
before the optimized step, while advancing the reference through 16 real slots.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import sys

import torch

PROJECT=Path(__file__).resolve().parents[1]
MEM=PROJECT/'results/small_official_20261010'
sys.path.insert(0,str(MEM/'code'))
sys.path.insert(0,str(MEM/'recovery_20261010'))
import recovery_runner as m
from fast_updater import updater_loss


def snapshot(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def metrics(a,b):
    assert a.keys()==b.keys()
    return {'exact':all(torch.equal(a[k],b[k]) for k in a),
            'max_abs':max([float((a[k]-b[k]).abs().max()) for k in a] or [0.]),
            'close':all(torch.allclose(a[k],b[k],rtol=1e-6,atol=1e-8) for k in a),
            'torch_fp32_close':all(torch.allclose(a[k],b[k],rtol=1.3e-6,atol=1e-5) for k in a)}


def run(panel,output):
    p=m.p
    if panel=='within':m.configure_within()
    manifest=m.read((m.W if panel=='within' else m.OLD)/'manifest.json')
    source=p.f.Store();rows=[r for r in manifest['rows'] if r['split']=='train']
    episodes=[p.episode(rows,0,i) for i in range(32)]
    ref,model,head,rule,sha=m.model_for(panel,'residual',0)
    model.train();model.checkpoint_blocks=True
    ck=torch.load(m.PROBES[panel],map_location='cpu',weights_only=False)
    probe=m.AttnProbe(ck['D'],ck['S']).cuda()
    probe.load_state_dict(ck['state_dict']);probe.eval().requires_grad_(False)
    initial=snapshot(model);probe_initial=snapshot(probe)
    frozen={k for k,v in model.named_parameters() if not v.requires_grad}
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4,weight_decay=.05)
    result={'complete':False,'panel':panel,'checkpoint_sha256':sha,'slots':[],
            'comparison':'same incoming weights and optimizer state at every slot',
            'rtol':1e-6,'atol':1e-8,
            'torch_fp32_tolerance':{'rtol':1.3e-6,'atol':1e-5},
            'code':{f:hashlib.sha256((PROJECT/'pipeline'/f).read_bytes()).hexdigest()
                    for f in ['fast_inference.py','fast_updater.py','benchmark_paired_updater.py']}}
    for slot in range(16):
        before=snapshot(model);optimizer_before=copy.deepcopy(opt.state_dict())
        states=[];grads=[];events=[]
        for fn in [p.updater_loss,updater_loss]:
            model.load_state_dict(before);opt.load_state_dict(copy.deepcopy(optimizer_before))
            opt.zero_grad(set_to_none=True);ee=[];active=False
            for ep in episodes[slot*2:slot*2+2]:
                loss,event=fn(model,probe,head,rule,source,ep,verify=True)
                ee.append(event)
                if loss.requires_grad:(loss/2).backward();active=True
            if active:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            grads.append({k:v.grad.detach().cpu().clone() for k,v in model.named_parameters() if v.grad is not None})
            states.append(snapshot(model));events.append(ee)
            if fn is p.updater_loss:reference_optimizer=copy.deepcopy(opt.state_dict())
        row={'slot':slot,'events_exact':events[0]==events[1],
             'gradients':metrics(*grads),'weights':metrics(*states),'events':events,
             'frozen_predictor_exact':all(torch.equal(s[k],initial[k]) for s in states for k in frozen)}
        result['slots'].append(row);m.write(output,result)
        print('PAIRED',panel,slot,{k:v for k,v in row.items() if k!='events'},flush=True)
        # Advance only the reference trajectory; the next pair has common inputs.
        model.load_state_dict(states[0]);opt.load_state_dict(reference_optimizer)
    result['probe_exact']=metrics(probe_initial,snapshot(probe))['exact']
    result['strict_complete']=result['probe_exact'] and all(r['events_exact'] and r['gradients']['close']
                          and r['weights']['close'] and r['frozen_predictor_exact'] for r in result['slots'])
    result['complete']=result['probe_exact'] and all(r['events_exact'] and r['gradients']['torch_fp32_close']
                          and r['weights']['torch_fp32_close'] and r['frozen_predictor_exact'] for r in result['slots'])
    m.write(output,result)
    assert result['complete'],str(output)
    print('PASS',panel,'paired steps',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--panel',choices=['cross','within'],required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(2);run(a.panel,a.output)
