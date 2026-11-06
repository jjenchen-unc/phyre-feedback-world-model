"""Matched A/B memory fine-tuning; fixed R0 initialization, base and plain L1."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from deploy_pilot_lib import PilotStore,episode
from redesign_lib import load_models,context_memory,write_json,digest

INIT='/mnt/meg/jjenchen/phyre-unc/runs/memory_redesign_20260928/R0/memory.pt'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--old-data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--variant',choices=['A','B'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=2000)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=json.loads((Path(a.root)/'paired_data.json').read_text())
    if not data['ab_ready']:
        write_json(out/'skipped.json',{'reason':'insufficient paired scenes; see pairing_audit.json'});return
    old=json.loads(Path(a.old_data).read_text());torch.set_num_threads(4);torch.manual_seed(a.seed)
    store=PilotStore();assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest()==data['files_sha256']
    base,probe,rule,pc,model=load_models(memory=INIT)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=3e-5,weight_decay=.05);start=0;log=[];counts=Counter()
    resume=out/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['args']==vars(a)
        model.load_state_dict(ck['weights'],strict=False);opt.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start=ck['step'];counts=Counter(ck['counts']);log=ck['log']
    write_json(out/'registration.json',{'args':vars(a),'initial_memory_sha256':digest(INIT),
        'paired_data_sha256':digest(Path(a.root)/'paired_data.json'),'loss':'unweighted future latent L1',
        'lr':3e-5,'legacy_replay_probability':.5,'base_frozen':True})
    t0=time.time();running=[]
    for step in range(start,a.steps):
        e=episode(data,old,store,step,a.seed,a.variant);q=e['query'];model.train();base.eval()
        z=torch.from_numpy(store.z(q)).cuda().float();act=torch.from_numpy(store.actions[q]).cuda()
        mem,mask=context_memory(model,store,e['context'])
        with torch.autocast('cuda',dtype=torch.bfloat16):pr=model(z[:,0],act,mem.expand(len(q),-1,-1),mask.expand(len(q),-1))
        loss=(pr.float()[:,1:]-z[:,1:]).abs().mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
        opt.step();counts[e['source']]+=1;counts['actual_k_'+str(len(e['context']))]+=1
        counts['queries']+=len(q);counts['positive_queries']+=int(store.labels[q].sum());running.append(float(loss.detach()))
        if (step+1)%100==0 or step+1==a.steps:
            r={'step':step+1,'loss':float(np.mean(running)),'grad_norm':float(norm),'seconds':round(time.time()-t0)}
            log.append(r);running=[];write_json(out/'progress.json',{'latest':r,'counts':dict(counts)});print(json.dumps(r),flush=True)
        if (step+1)%500==0 or step+1==a.steps:
            weights={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('base.')}
            ck={'state_dict':weights,'mode':'residual','mem_steps':16,'mem_grid':10,'per_attempt':True,
                'gate_empty':True,'unfreeze':False,'base':str(__import__('redesign_lib').BASE),'args':vars(a),'step':step+1}
            torch.save(ck,out/'memory.tmp.pt');(out/'memory.tmp.pt').replace(out/'memory.pt')
            torch.save({'args':vars(a),'weights':weights,'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'step':step+1,'counts':dict(counts),'log':log},out/'resume.tmp.pt')
            (out/'resume.tmp.pt').replace(resume)
    write_json(out/'training_complete.json',{'steps':a.steps,'counts':dict(counts),'memory_sha256':digest(out/'memory.pt')})


if __name__=='__main__':main()
