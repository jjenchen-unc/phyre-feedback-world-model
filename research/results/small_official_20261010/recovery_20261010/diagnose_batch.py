import sys,json
from pathlib import Path
import torch,numpy as np
R=Path('/path/to/phyre-unc/results/small_official_20261010');sys.path.insert(0,str(R/'code'))
from encoder import Encoder
from run_small import PCA
enc=Encoder(PCA);torch.set_num_threads(4)
raw=next((R/'assets/cross/observations').glob('.raw*.npz'));z=np.load(raw);task=str(z['task']);a=z['action'];c=np.load(R/'assets/cross/candidates'/(task.replace(':','_')+'.npz'));i=int(np.flatnonzero((c['action']==a).all(1))[0]);start=i//32*32
s=np.load(R/'assets/cross/shortlist'/(task.replace(':','_')+'.npz'));j=int(np.flatnonzero((s['action']==a).all(1))[0])
frames=np.concatenate([z['frames'][:1],c['frames'][:31]])
truth=enc.rollout(z['frames'])[0];res={}
for b in [1,2,4,8,16,32]:
 v=enc.initial(frames[:b])[0];d=np.abs(v.astype(float)-truth);res[str(b)]={'max':float(d.max()),'mean':float(d.mean())}
 print(b,res[str(b)],flush=True)
with torch.no_grad():
 x=enc.pixels(frames[:8]);v=x[:,None].expand(-1,16,-1,-1,-1).contiguous();h=enc.model(pixel_values=v.permute(0,2,1,3,4)).last_hidden_state[:,1:]
 lat=enc.project(h.reshape(8,16,196,-1)[:,0]).cpu().numpy();d=np.abs(lat[0].astype(float)-truth);res['b8t16']={'max':float(d.max()),'mean':float(d.mean())}
print(res,flush=True)
(R/'recovery_20261010/BATCH_DIAG.json').write_text(json.dumps(res,indent=2))
