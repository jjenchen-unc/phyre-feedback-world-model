"""Evaluate every pilot model under identical random and policy contexts.

Fixed-query evaluation only: this is not a full adaptive 1024-action rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from deploy_pilot_lib import PilotStore
from redesign_lib import load_models,context_memory,readout,write_json,digest


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--memory',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.root);data=json.loads((root/'paired_data.json').read_text());torch.set_num_threads(4)
    if not data['ab_ready']:write_json(a.out,{'skipped':True,'reason':'insufficient paired scenes'});return
    store=PilotStore();assert hashlib.sha256('\n'.join(store.files).encode()).hexdigest()==data['files_sha256']
    base,probe,rule,_,model=load_models(memory=a.memory);rows=[];tasks=data['eligible_validation_tasks']
    for task in tasks:
        for e in data['tasks'][task]['episodes']:
            donor=next((x['B'] for t in tasks if t[:5]!=task[:5] for x in data['tasks'][t]['episodes'] if len(x['B'])==e['actual_k']),None)
            if donor is None:donor=next((x['B'] for t in tasks if t!=task for x in data['tasks'][t]['episodes'] if len(x['B'])==e['actual_k']),None)
            conditions=[('base',[]),('empty',[]),('random_context',e['A']),('policy_context',e['B'])]
            if donor is not None:conditions.append(('wrong_scene',donor))
            base_scores=None
            for condition,ctx in conditions:
                q=e['query'];y=store.labels[q];scs=[];losses=[]
                mem=context_memory(model,store,ctx) if condition!='base' else None
                for i in range(0,len(q),16):
                    ids=q[i:i+16];z=torch.from_numpy(store.z(ids)).cuda().float();act=torch.from_numpy(store.actions[ids]).cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=base(z[:,0],act) if condition=='base' else model(z[:,0],act,mem[0].expand(len(ids),-1,-1),mem[1].expand(len(ids),-1))
                    pred=pred.float();sc=readout(pred,[store.feature(j) for j in ids],probe,rule)
                    scs.extend(sc.cpu().tolist());losses.extend((pred[:,1:]-z[:,1:]).abs().mean((1,2,3)).cpu().tolist())
                if condition=='base':base_scores=scs
                if condition=='empty':assert np.array_equal(scs,base_scores)
                ordered=y[np.argsort(-np.array(scs),kind='stable')]
                rows.append({'task':task,'attempt':e['attempt'],'actual_k':e['actual_k'],'condition':condition,
                    'context':ctx,'query':q,'rank_auc':float(roc_auc_score(y,scs)) if y.any() and (~y).any() else None,
                    'first_success_rank':int(np.flatnonzero(ordered)[0])+1 if y.any() else None,
                    'per_query':[{'index':int(j),'solved':bool(yy),'score':s,'latent_l1':l} for j,yy,s,l in zip(q,y,scs,losses)]})
        print(json.dumps({'task':task,'rows':len(rows)}),flush=True)
    write_json(a.out,{'memory':a.memory,'memory_sha256':digest(a.memory),'paired_data_sha256':digest(root/'paired_data.json'),
        'scope':'fixed common query targets, memory-only held templates, conditional on matched nonempty prefixes; no cross/deployment success claim',
        'n_tasks':len(tasks),'rows':rows})


if __name__=='__main__':main()
