"""Matched numerical audit of initial-only vs recorded-first-token policy inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from redesign_lib import load_models,readout,write_json


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    d=Path('/path/to/phyre-unc/results/memory_deploy_pilot_20260929')
    m=Path('/mnt/meg/jjenchen/phyre-unc/deploy_pilot_20260929')
    tr=json.loads((d/'traces/00001_000.json').read_text());rows=tr['manifest']
    with np.load(m/'scores/00001_000.npz') as p:
        local=[r['local'] for r in rows];actions=p['action'][local];initial=p['initial_latent'][local]
        f0=p['feats0'][local];cached=p['scores'][local]
    recorded=np.stack([np.load(m/'latents/levjepa_overlap_g32_pca128'/Path(r['file']).with_suffix('.npy').name)[0] for r in rows])
    base,probe,rule,_,_=load_models();sc={}
    for name,inputs in [('initial_only',initial),('recorded_first',recorded)]:
        vals=[]
        for i in range(0,len(rows),16):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=base(torch.from_numpy(inputs[i:i+16]).cuda().float(),torch.from_numpy(actions[i:i+16]).cuda()).float()
            vals.extend(readout(pred,[f[None] for f in f0[i:i+16]],probe,rule).cpu().tolist())
        sc[name]=np.array(vals)
    aa,bb=sc['initial_only'],sc['recorded_first'];order_a=np.argsort(-aa,kind='stable');order_b=np.argsort(-bb,kind='stable')
    result={'scope':'one preselected smoke scene, 42 already collected actions; not a full-shortlist agreement estimate; collection policy unchanged',
        'n':len(rows),'same_input_cached_score_max_abs':float(np.abs(aa-cached).max()),
        'input_route_score_max_abs':float(np.abs(aa-bb).max()),'input_route_score_mean_abs':float(np.abs(aa-bb).mean()),
        'spearman':float(spearmanr(aa,bb).statistic),'top10_overlap':len(set(order_a[:10]) & set(order_b[:10])),
        'same_top1':bool(order_a[0]==order_b[0]),
        'per_action':[{'action_idx':r['action_idx'],'solved':r['solved'],'initial_score':float(x),'recorded_score':float(y)} for r,x,y in zip(rows,aa,bb)]}
    write_json(d/'initial_route_score_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='per_action'}),flush=True)


if __name__=='__main__':main()
