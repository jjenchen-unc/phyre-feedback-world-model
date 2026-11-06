"""Fixed-query evidence utility panel on memory-held-out training templates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from redesign_lib import Store, load_models, context_memory, readout, write_json, digest


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True)
    p.add_argument('--memory', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); torch.set_num_threads(4)
    data = json.loads(Path(args.data).read_text()); store = Store()
    base, probe, rule, _, model = load_models(memory=args.memory)
    tasks = data['validation_tasks']; rows = []; t0 = time.time()
    for ti, task in enumerate(tasks):
        d = data['tasks'][task]; q = d['query_pool']
        support = d['base_trace']['failures']
        # Keep even zero-failure cases; report effective K and nonempty subgroups.
        for requested_k in [1, 4]:
            ctx = support[:requested_k]
            choices = tasks[ti+1:] + tasks[:ti]
            donor_task = next((t for t in choices if len(data['tasks'][t]['base_trace']['failures']) >= len(ctx)), None)
            donor = data['tasks'][donor_task]['base_trace']['failures'] if donor_task else []
            for name, context in [('base', []), ('empty', []), ('correct', ctx), ('wrong_scene', donor[:len(ctx)])]:
                if name == 'wrong_scene' and len(donor) < len(ctx): continue
                mem = context_memory(model, store, context) if name != 'base' else None
                values, score, green = [], [], []
                for start in range(0, len(q), 16):
                    ids = q[start:start+16]
                    truth = torch.from_numpy(store.z(ids)).cuda().float()
                    a = torch.from_numpy(store.actions[ids]).cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pr = (base(truth[:,0], a) if name == 'base' else model(truth[:,0], a, mem[0].expand(len(ids),-1,-1), mem[1].expand(len(ids),-1))).float()
                    fs = [store.feature(i) for i in ids]
                    sc, xy, _ = readout(pr, fs, probe, rule, details=True)
                    values.extend((pr[:,1:]-truth[:,1:]).abs().mean((1,2,3)).cpu().tolist())
                    score.extend(sc.cpu().tolist())
                    for j,f in enumerate(fs):
                        g = int(np.flatnonzero(f[0,:,9]>.5)[0])
                        green.append(float(np.median(np.linalg.norm(xy[j,1:].cpu().numpy()-f[1:,g,:2]*256,axis=-1))))
                y = store.labels[q]; ordered = y[np.argsort(-np.asarray(score),kind='stable')]
                per = [{'index':int(i),'solved':bool(yy),'latent_l1':v,'score':s,'green_error_px':g}
                       for i,yy,v,s,g in zip(q,y,values,score,green)]
                rows.append({'task':task,'requested_k':requested_k,'actual_k':len(ctx),
                             'condition':name,'context':context,'donor_k_matches':len(donor)>=len(ctx),
                             'first_success_rank':int(np.flatnonzero(ordered)[0])+1 if y.any() else 0,
                             'rank_auc':float(roc_auc_score(y,score)) if y.any() and (~y).any() else None,
                             'per_query':per})
        if ti%8==0: print(json.dumps({'tasks':ti+1,'seconds':round(time.time()-t0)}),flush=True)
    write_json(args.out,{'memory':args.memory,'memory_sha256':digest(args.memory),'data_sha256':digest(args.data),
                          'scope':data['holdout_scope'],'n_tasks':len(tasks),'rows':rows})
    print('FIXED_EVAL_DONE',flush=True)


if __name__=='__main__': main()
