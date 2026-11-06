import sys,json
from pathlib import Path
import torch,numpy as np
R=Path('/path/to/phyre-unc/results/small_official_20261010');sys.path.insert(0,str(R/'code'))
from encoder import Encoder
from run_small import PCA
enc=Encoder(PCA);torch.set_num_threads(4)
raw=next((R/'assets/cross/observations').glob('.raw*.npz'));z=np.load(raw);task=str(z['task']);a=z['action'];c=np.load(R/'assets/cross/candidates'/(task.replace(':','_')+'.npz'));i=int(np.flatnonzero((c['action']==a).all(1))[0]);start=i//32*32
s=np.load(R/'assets/cross/shortlist'/(task.replace(':','_')+'.npz'));j=int(np.flatnonzero((s['action']==a).all(1))[0])
vals={'shortlist':s['z0'][j],'single':enc.initial(z['frames'][:1])[0],'batch':enc.initial(c['frames'][start:start+32])[i-start],'full':enc.rollout(z['frames'])[0]}
# Repeat initial-only clip to the exact 16-frame / batch8 shape used by old full rollout encoding.
with torch.no_grad():
 x=enc.pixels(z['frames'][:1]);v=x[None].expand(8,16,3,224,224).contiguous();h=enc.model(pixel_values=v.permute(0,2,1,3,4)).last_hidden_state[:,1:]
 vals['repeat16']=enc.project(h.reshape(8,16,196,-1)[0,0]).cpu().numpy()
r={}
for k,v in vals.items():
 d=np.abs(v.astype(float)-vals['full']);r[k]={'max':float(d.max()),'mean':float(d.mean())}
print(json.dumps(r,indent=2),flush=True)
(R/'recovery_20261010/ENCODER_DIAG.json').write_text(json.dumps(r,indent=2))
