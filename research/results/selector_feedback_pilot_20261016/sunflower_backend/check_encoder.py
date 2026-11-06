"""Compare A5000 observation encodings with existing canonical A6000 targets."""
import os,sys,json,time
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parent
os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
mode=int(sys.argv[1])
if mode:os.environ['DISABLE_ADDMM_CUDA_LT']='1'
else:os.environ.pop('DISABLE_ADDMM_CUDA_LT',None)
sys.path.insert(0,str(R/'code'))
import numpy as np
import torch
import selector_feedback_data as pools
torch.set_num_threads(2)
plan=pools.data.checked();enc=pools.data.runtime().m.Encoder(pools.data.runtime().m.PCA)
worker=pools.data.CPUWorker();records=[]
try:
 for panel in plan['panels']:
  paths=sorted((R/'observations'/panel).glob('*/*.npz'))[:3]
  assert len(paths)==3
  for i,p in enumerate(paths):
   with np.load(p) as z:
    task=str(z['task']);action=z['action'].tolist();expected=z['latent'].copy()
   raw=B/f'check_{mode}_{panel}_{i}.npz'
   worker.request(mode='rollout',task=task,action=action,out=str(raw))
   with np.load(raw) as z:
    actual=enc.rollout(z['frames']);initial=enc.initial(z['frames'][:1])[0]
   pool=pools.data.load_pool(panel,task)
   matches=np.flatnonzero(np.all(pool['simulation_action']==action,axis=1));assert len(matches)==1
   d=np.abs(actual.astype(float)-expected.astype(float))
   records.append({'panel':panel,'task':task,'reference':str(p),'reference_sha256':pools.sha(p),
     'exact':bool(np.array_equal(actual,expected)),'initial_exact':bool(np.array_equal(initial,pool['z0'][matches[0]])),
     'max_abs':float(d.max()),'mean_abs':float(d.mean())})
   raw.unlink();print(records[-1],flush=True)
finally:worker.close()
pools.write(B/f'ENCODER_CHECK_{mode}.json',{'records':records,'all_exact':all(r['exact'] and r['initial_exact'] for r in records),'gpu':torch.cuda.get_device_name(0),'disable_addmm':mode})
