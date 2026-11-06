"""Differentiable counterpart of the frozen geometry/log-contact scorer.

Initial scene metadata are constants. Probe parameters may be frozen while
input gradients remain enabled. No learned head or changed scoring rule here.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def scores_from_distances(distance, tau=4., scale=1.):
    return torch.logsumexp(F.logsigmoid((tau-distance.double())/scale),dim=-1)


def distances(probe,prediction,features):
    b,nt,ns,_=prediction.shape
    with torch.autocast(prediction.device.type,enabled=False):
        kv=probe.inp(prediction.float().reshape(b*nt,ns,-1))+probe.space
        rows=[]
        for j,f in enumerate(features):
            f0=np.asarray(f)[0]; xy=[];angles=[]
            for o in range(len(f0)):
                if f0[o,13]>=.5 or f0[o,11]>=.5:
                    xy.append(torch.as_tensor(f0[o,:2],device=prediction.device).expand(nt,2)*256)
                    angles.append(torch.full((nt,),float(f0[o,2]),device=prediction.device)*2*math.pi)
                else:
                    q=torch.as_tensor(np.repeat(f0[None,o],nt,axis=0),device=prediction.device)
                    p,a=probe.read(kv[j*nt:(j+1)*nt],q)
                    # Match the reference feature round-trip before geometry.
                    xy.append((p/256)*256)
                    angles.append(((torch.atan2(a[:,1],a[:,0])/2)/(2*math.pi)%1)*(2*math.pi))
            g=int(np.flatnonzero(f0[:,9]>.5)[0]);goals=np.flatnonzero((f0[:,10]>.5)|(f0[:,11]>.5))
            cg=xy[g];rg=float(f0[g,3]*256/2);ds=[]
            for o in goals:
                co=xy[o];length=float(f0[o,3]*256)
                if int(f0[o,4:8].argmax())==1:
                    u=torch.stack([angles[o].cos(),angles[o].sin()],dim=-1)
                    along=((cg-co)*u).sum(-1).clamp(-length/2,length/2)
                    ds.append(torch.linalg.vector_norm(cg-(co+along[:,None]*u),dim=-1)-rg)
                else:
                    ds.append(torch.linalg.vector_norm(cg-co,dim=-1)-rg-length/2)
            rows.append(torch.stack(ds).amin(0))
        return torch.stack(rows)


def score(probe,prediction,features):
    return scores_from_distances(distances(probe,prediction,features))


def supervised_terms(scores,labels,calibration):
    y=torch.as_tensor(labels,device=scores.device,dtype=torch.bool)
    logits=calibration['slope']*scores+calibration['intercept']
    bce=F.binary_cross_entropy_with_logits(logits,y.double())
    if y.any() and (~y).any():
        pair=F.softplus(logits[~y][None,:]-logits[y][:,None]).mean()
    else:
        pair=scores.sum()*0
    return bce,pair
