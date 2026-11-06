"""Pinned legacy LeVJEPA/PCA preprocessing; cached observations, never residuals."""
import torch
from torch.nn import functional as F
import numpy as np
from transformers import AutoModel
class Encoder:
 def __init__(self,pca):
  self.model=AutoModel.from_pretrained('galilai-group/LeVJEPA-VideoMix-Large',trust_remote_code=True,
   revision='e831a0347737fcaa660b39c57d41c109de399845',local_files_only=True).eval().cuda().half()
  self.palette=torch.tensor([[255,255,255],[243,79,70],[107,206,187],[24,119,242],[75,74,164],[185,202,210],[0,0,0],[252,223,227]],device='cuda',dtype=torch.float32)/255
  self.mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);self.std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
  p=torch.load(pca,map_location='cpu',weights_only=False);self.pm=p['mean'].cuda();self.pc=p['components'].cuda()
 def pixels(self,codes):
  x=self.palette[torch.from_numpy(codes.astype(np.int64)).cuda()].flip(1).permute(0,3,1,2)
  x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
  return ((x-self.mean)/self.std).half()
 def project(self,h):return ((h.float()-self.pm)@self.pc).half()
 @torch.no_grad()
 def initial(self,codes):
  x=self.pixels(codes);clip=torch.stack([x,x],2)
  h=self.model(pixel_values=clip).last_hidden_state[:,1:197]
  return self.project(h).cpu().numpy()
 @torch.no_grad()
 def rollout(self,frames):
  x=self.pixels(frames);clips=torch.stack([x[s:s+16] for s in range(0,113,8)])
  parts=[]
  for start in range(0,len(clips),8):
   v=clips[start:start+8];h=self.model(pixel_values=v.permute(0,2,1,3,4)).last_hidden_state[:,1:]
   parts.append(h.reshape(len(v),16,196,-1))
  h=torch.cat(parts);steps=torch.cat([h[0]]+[v[-8:] for v in h[1:]])
  assert len(steps)==128
  return self.project(steps[list(range(0,128,4))]).cpu().numpy()
