"""Real-data causal-policy and frozen-predictor BPTT smoke; not formal training."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent;P=R.parent/'query_token_dynamic_20261005'
sys.path.insert(0,str(P/'code'));sys.path.insert(0,str(R/'code'))
import fresh_memory_geometry as f
from prefix_policy_predictor import PrefixPredictor,policy_rollout,replay_state

def main():
    torch.set_num_threads(4);torch.manual_seed(20261008)
    old=f.read(R.parent/'query_token_dynamic_full_eval_20261007/SUMMARY.json')
    assert old['complete'] and old['raw_audit_passed']
    config=f.read(R/'PREPARATION.json')
    assert all(f.digest(R/'code'/name)==sha for name,sha in config['code'].items())
    m=f.read(P/'manifest.json');assert f.digest(P/'manifest.json')==config['parent_manifest_sha256']
    ref,probe,rule,pc,_=f.load_models();heads,hashes=f.load_heads(m,f.PROBES,probe)
    assert hashes[m['primary_head']]==m['heads'][m['primary_head']]
    head=heads[m['primary_head']];model=PrefixPredictor(ref,pc['Din'],stage='updater').cuda().eval()
    source=f.Store();frozen=f.frozen_fingerprint(model.base);initial=model.initial_tokens.detach().clone()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    rows=sorted([r for r in config['rows'] if r['split']=='train'],key=lambda r:f.stable(('prefix_smoke',r['task'])))[:32]
    events=[];steps=0;start=time.time();torch.cuda.reset_peak_memory_stats()
    for row in rows:
        ids=row['candidates'];z0=torch.from_numpy(np.stack([source.z([i])[0,0] for i in ids])).cuda().float()
        actions=torch.from_numpy(source.actions[ids]).cuda();initial_features=[source.feature(i)[:1].copy() for i in ids]
        observed=[]
        def score_fn(pred,features):
            with torch.autocast('cuda',enabled=False):v=f.readout(head,pred.float(),features,rule)['log_score']
            return torch.from_numpy(v).to(pred.device)
        def observe(local):
            observed.append(local)
            return bool(source.labels[ids[local]]),torch.from_numpy(source.z([ids[local]])).cuda().float()
        def truth_for(local):return torch.from_numpy(source.z([ids[local]])).cuda().float()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            trace,supports=policy_rollout(model,z0,actions,initial_features,score_fn,observe,budget=4)
        assert len(observed)==len(trace) and len(set(observed))==len(observed)
        assert not any(t['solved'] for t in trace[:-1])
        event={'task':row['task'],'selected':[ids[t['local']] for t in trace],
               'solved':[t['solved'] for t in trace],'updates':len(supports)}
        if supports:
            with torch.autocast('cuda',dtype=torch.bfloat16):state=replay_state(model,z0,actions,supports,truth_for)
            assert torch.equal(state.detach().cpu(),trace[-1]['state'])
            qi=row['query'][:2];z=torch.from_numpy(source.z(qi)).cuda().float();qa=torch.from_numpy(source.actions[qi]).cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):before=model(z[:,0],qa).float()
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z[:,0],qa,state.expand(len(qi),-1,-1)).float()
            loss=(pred[:,1:]-z[:,1:]).abs().mean()+.1*f.geometry_loss(probe,pred,[source.feature(i) for i in qi],row['times'][:2])
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(loss) and torch.isfinite(norm) and norm>0
            opt.step();steps+=1
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=model(z[:,0],qa).float()
            assert torch.equal(before,after) and torch.equal(model.initial_tokens,initial)
            assert f.frozen_fingerprint(model.base)==frozen
            event.update(loss=float(loss.detach()),exact_current_state_replay=True,frozen_Q0_prediction=True)
        events.append(event);print('SMOKE_EPISODE',json.dumps(event),flush=True)
        if steps>=3:break
    assert steps==3,'Need actual scorer-selected failures to test updater gradients'
    f.write_json(R/'SMOKE.json',{'passed':True,'scope':'Only common prefix/updater implementation; fresh smoke parameters discarded',
        'preparation_sha256':f.digest(R/'PREPARATION.json'),'optimizer_steps':steps,'events':events,
        'seconds':time.time()-start,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,
        'formal_training_submitted':False})
    print('PREFIX_POLICY_SMOKE_PASS',flush=True)

if __name__=='__main__':main()
