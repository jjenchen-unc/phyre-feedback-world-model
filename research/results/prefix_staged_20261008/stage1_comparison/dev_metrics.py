"""Matched original versus final Q0 on the same historical 400-trajectory dev pack."""
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
import prefix_staged as p
f=p.f
@torch.no_grad()
def main():
 torch.set_num_threads(4);m=p.checked(R);prov=f.provenance(R)
 source=f.Pack(m['stage1_dev_pack']);assert len(source.actions)==400
 ref,probe,rule,pc,_=f.load_models();ref.eval();probe.eval()
 rows={};checkpoints={};start=time.time()
 for name in ['original','q0_s0','q0_s1','q0_s2']:
  if name=='original':model=ref;checkpoints[name]=m['base_sha256']
  else:
   seed=int(name[-1]);done=f.read(R/'models'/name/'complete.json')
   assert done['provenance']==prov and done['slots']==52272
   path=Path(done['checkpoint']);assert f.digest(path)==done['checkpoint_sha256'];checkpoints[name]=done['checkpoint_sha256']
   ck=torch.load(path,map_location='cpu',weights_only=False)
   model=p.PrefixPredictor(copy.deepcopy(ref),pc['Din'],stage='initial',**m['architecture']).cuda()
   model.load_state_dict(ck['state_dict']);model.eval()
  l1=[];geometry=0
  for startidx in range(0,400,8):
   ids=list(range(startidx,min(startidx+8,400)));z=torch.from_numpy(source.z(ids)).cuda().float();act=torch.from_numpy(source.actions[ids]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z[:,0].contiguous(),act).float()
   l1.extend((pred[:,1:]-z[:,1:]).abs().mean((1,2,3)).cpu().tolist())
   geometry+=float(f.geometry_loss(probe,pred,[source.feature(i) for i in ids],p.old.target_times(ids)))*len(ids)
  rows[name]={'n':400,'latent_l1':float(np.mean(l1)),'geometry_loss':geometry/400,'per_trajectory_l1':l1}
  if name!='original':
   saved=f.read(R/'models'/name/'dev_052272.json')['latent_l1'];assert abs(saved-rows[name]['latent_l1'])<1e-8
   del model,ck
  print(name,{k:v for k,v in rows[name].items() if k!='per_trajectory_l1'},flush=True)
 f.write_json(R/'stage1_comparison/DEV_METRICS.json',{'complete':True,'provenance':prov,'checkpoint_sha256':checkpoints,
  'dev_pack':m['stage1_dev_pack'],'meta_sha256':m['stage1_dev_meta_sha256'],'source_script_sha256':f.digest(__file__),
  'scope':'Same400 historical cross-template dev trajectories; NOT full1024 ranking pools or within pack; no memory updates',
  'geometry_definition':'Original frozen probe, same deterministic8times per trajectory, equal-query dynamic-object loss used for training',
  'models':rows,'seconds':time.time()-start})
if __name__=='__main__':main()
