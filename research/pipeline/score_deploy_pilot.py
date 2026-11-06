"""Deployment candidate scoring: initial-image encoder, top-1024, per-query rerank."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from redesign_lib import load_models, write_json
from common import surface_dist, rule_score


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',required=True);p.add_argument('--out',required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(4);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    enc=AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large',trust_remote_code=True,
        revision='e831a0347737fcaa660b39c57d41c109de399845').eval().cuda().half()
    base,probe,rule,_,_=load_models()
    pc=torch.load('/mnt/meg/jjenchen/phyre-unc/latents/levjepa_overlap_g32_pca128/pca.pt',weights_only=False)
    mu,comp=pc['mean'].cuda(),pc['components'].cuda()
    palette=torch.tensor([[255,255,255],[243,79,70],[107,206,187],[24,119,242],
        [75,74,164],[185,202,210],[0,0,0],[252,223,227]],device='cuda').float()/255
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    def scores(pr, f0, vectorized):
        b,nt,ns,_=pr.shape;kv=probe.inp(pr.float().reshape(b*nt,ns,-1))+probe.space
        fs=np.repeat(f0[:,None],nt,axis=1).copy()
        dyn=[o for o in range(f0.shape[1]) if f0[0,o,13]<.5 and f0[0,o,11]<.5]
        if vectorized:
            for o in dyn:
                q=torch.from_numpy(np.repeat(f0[:,o],nt,axis=0)).cuda()
                xy,angle=probe.read(kv,q);xy=xy.reshape(b,nt,2).cpu().numpy();angle=angle.reshape(b,nt,2).cpu().numpy()
                fs[:,:,o,:2]=xy/256;fs[:,:,o,2]=np.arctan2(angle[...,1],angle[...,0])/(4*np.pi)%1
        else:
            for j in range(b):
                for o in dyn:
                    q=torch.from_numpy(np.repeat(f0[j:j+1,o],nt,axis=0)).cuda()
                    xy,angle=probe.read(kv[j*nt:(j+1)*nt],q);angle=angle.cpu().numpy()
                    fs[j,:,o,:2]=xy.cpu().numpy()/256
                    fs[j,:,o,2]=np.arctan2(angle[:,1],angle[:,0])/(4*np.pi)%1
        return [rule_score(surface_dist(f),rule) for f in fs]
    t0=time.time()
    for path in sorted(Path(a.candidates).glob('*.npz'))[a.shard::a.nshards]:
        target=out/path.name
        if target.exists():continue
        with np.load(path) as c:
            f0=c['feats0'];actions=c['action'];codes=c['frames'];first=[];initial=[]
            for i in range(0,len(codes),32):
                x=palette[torch.from_numpy(codes[i:i+32].astype(np.int64)).cuda()].flip(1).permute(0,3,1,2)
                x=((F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)-mean)/std).half()
                z=enc(pixel_values=torch.stack([x,x],2)).last_hidden_state[:,1:197]
                z=((z.half().float()-mu)@comp).half();initial.append(z.cpu().numpy())
                with torch.autocast('cuda',dtype=torch.bfloat16):pr=base(z.float(),torch.from_numpy(actions[i:i+32]).cuda()).float()
                first.extend(scores(pr,f0[i:i+32],True))
            initial=np.concatenate(initial);first=np.array(first);top=np.argsort(-first,kind='stable')[:1024]
            second=[]
            for i in range(0,len(top),32):
                ii=top[i:i+32]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pr=base(torch.from_numpy(initial[ii]).cuda().float(),torch.from_numpy(actions[ii]).cuda()).float()
                second.extend(scores(pr,f0[ii],False))
            tmp=target.with_suffix('.tmp.npz')
            # Outcomes never influence either rank; retained for offline audit only.
            np.savez_compressed(tmp,action_idx=c['action_idx'][top],action=actions[top],feats0=f0[top],
                initial_latent=initial[top],scores=np.array(second),stage1_scores=first[top],
                cache_status=c['status'][top],n_valid_candidates=len(first),
                full_action_idx=c['action_idx'],full_scores=first)
            tmp.replace(target)
            print(json.dumps({'task':path.stem,'valid':len(first),'shortlist':len(top),'seconds':round(time.time()-t0)}),flush=True)
    write_json(out/f'complete_{a.shard}.json',{'complete':True,'seconds':round(time.time()-t0)})


if __name__=='__main__':main()
