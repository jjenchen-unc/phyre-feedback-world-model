"""Independent free-running updater histories with strict numerical controls."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
MEM=PROJECT/'results/small_official_20261010'
sys.path.insert(0,str(MEM/'code'))
sys.path.insert(0,str(MEM/'recovery_20261010'))
import recovery_runner as m
import fast_updater
from fast_inference import PreparedScorer
from updater_validation_numerics import strict_except_disjoint_pool


def fingerprint(items):
    h=hashlib.sha256()
    for name,value in items:
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run(panel,output,slots,repeats,audit):
    if output.exists(): raise FileExistsError(output)
    if panel=='within':m.configure_within()
    p=m.p
    manifest=m.read((m.W if panel=='within' else m.OLD)/'manifest.json')
    source=p.f.Store();rows=[r for r in manifest['rows'] if r['split']=='train']
    episodes=[p.episode(rows,0,i) for i in range(2*slots)]
    ref,model,head,rule,ck_sha=m.model_for(panel,'residual',0)
    model.train();model.checkpoint_blocks=True
    ck=torch.load(m.PROBES[panel],map_location='cpu',weights_only=False)
    probe=m.AttnProbe(ck['D'],ck['S']).cuda()
    probe.load_state_dict(ck['state_dict']);probe.eval().requires_grad_(False)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    frozen_before=fingerprint((k,v) for k,v in model.named_parameters() if not v.requires_grad)
    probe_before=fingerprint(probe.state_dict().items())
    code={name:hashlib.sha256((PROJECT/'pipeline'/name).read_bytes()).hexdigest()
          for name in ['fast_inference.py','fast_updater.py','validate_updater_trajectory.py','updater_validation_numerics.py']}
    result={'complete':False,'panel':panel,'slots':slots,'checkpoint_sha256':ck_sha,'code':code,
            'episode_sha256':hashlib.sha256(json.dumps(episodes,sort_keys=True).encode()).hexdigest(),
            'numerics':'strict deterministic kernels except unchanged disjoint adaptive-pool backward',
            'shared_gpu_timing':True,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'audit_scorer':audit,'arms':[]}
    audit_counts={'chunks':0,'scores':0}
    class AuditedScorer(PreparedScorer):
        @torch.no_grad()
        def __call__(self,pred,features):
            got=super().__call__(pred,features)
            expected=p.score_function(self.probe,self.rule)(pred,features)
            assert torch.equal(got,expected),'Scorer differs at identical current prediction'
            audit_counts['chunks']+=1;audit_counts['scores']+=len(got)
            return got
    fast_updater.PreparedScorer=AuditedScorer if audit else PreparedScorer
    arms=[('reference',p.updater_loss),('optimized',fast_updater.updater_loss)]
    if repeats:arms.append(('reference_repeat',p.updater_loss))
    with strict_except_disjoint_pool() as control:
        for name,fn in arms:
            model.load_state_dict(initial)
            torch.manual_seed(1701);torch.cuda.manual_seed_all(1701)
            opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4,weight_decay=.05)
            arm={'name':name,'events':[],'gradient_sha256':[],'weight_sha256':[],'seconds':[]}
            for slot in range(slots):
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize();start=time.perf_counter()
                events=[];active=False
                for ep in episodes[2*slot:2*slot+2]:
                    loss,event=fn(model,probe,head,rule,source,ep,verify=True)
                    events.append(event)
                    if loss.requires_grad:(loss/2).backward();active=True
                if active:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                torch.cuda.synchronize();arm['seconds'].append(time.perf_counter()-start)
                arm['events'].append(events)
                arm['gradient_sha256'].append(fingerprint((k,v.grad) for k,v in model.named_parameters() if v.grad is not None))
                arm['weight_sha256'].append(fingerprint((k,v) for k,v in model.named_parameters() if v.requires_grad))
                print('SLOT',panel,name,slot,[e['used_updates'] for e in events],flush=True)
                result['progress']={'arm':name,'slots':slot+1};m.write(output,result)
            arm['frozen_predictor_exact']=frozen_before==fingerprint((k,v) for k,v in model.named_parameters() if not v.requires_grad)
            arm['probe_exact']=probe_before==fingerprint(probe.state_dict().items())
            result['arms'].append(arm);m.write(output,result)
            del opt;gc.collect();torch.cuda.empty_cache()
        result['pool_backward_calls']=control.calls
        result['pool_shapes']=sorted(control.shapes)
    result['comparisons']=[]
    baseline=result['arms'][0]
    for other in result['arms'][1:]:
        result['comparisons'].append({'arm':other['name'],
            'events_exact':baseline['events']==other['events'],
            'gradients_exact':baseline['gradient_sha256']==other['gradient_sha256'],
            'weights_exact':baseline['weight_sha256']==other['weight_sha256'],
            'first_differing_event_slot':next((i for i,(a,b) in enumerate(zip(baseline['events'],other['events'])) if a!=b),None)})
    result['audited_scores']=audit_counts
    result['complete']=all(c['events_exact'] and c['gradients_exact'] and c['weights_exact'] for c in result['comparisons']) and all(a['frozen_predictor_exact'] and a['probe_exact'] for a in result['arms'])
    result.pop('progress',None);m.write(output,result)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='arms'}),flush=True)
    assert result['complete'],str(output)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--panel',choices=['cross','within'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--slots',type=int,default=16)
    ap.add_argument('--reference-repeat',action='store_true');ap.add_argument('--audit-scorer',action='store_true')
    a=ap.parse_args();torch.set_num_threads(2)
    if a.slots<1:ap.error('--slots must be positive')
    run(a.panel,a.output,a.slots,a.reference_repeat,a.audit_scorer)
