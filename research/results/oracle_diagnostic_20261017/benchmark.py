"""Cold-observation end-to-end oracle cost pilot; no method conclusion or training."""
from pathlib import Path
import sys,os,time,json,argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
P=Path('/path/to/phyre-unc');R=P/'results/oracle_diagnostic_20261017';F=P/'results/selector_feedback_pilot_20261016'
sys.path.insert(0,str(F/'code'));sys.path.insert(0,str(F/'sunflower_backend'))
from encoder_service import request
import selector_feedback_run as run
import torch
ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);a=ap.parse_args()
torch.set_num_threads(2)
plan=run.data.checked();task=plan['panels'][a.panel]['tasks'][0];pool=run.data.load_pool(a.panel,task)
team=run.Team(a.panel,'q0',plan);worker=run.data.CPUWorker();folder=R/'benchmark'/a.panel;folder.mkdir(parents=True,exist_ok=True)
latents=[];features=[];records=[]
def raw(i):
 start=time.perf_counter();p=folder/f'raw_{i}.npz'
 response=worker.request(mode='rollout',task=task,action=pool['simulation_action'][i].tolist(),out=str(p))
 assert response['status']==int(pool['status'][i])
 return p,time.perf_counter()-start
start=time.perf_counter()
try:
 with ThreadPoolExecutor(max_workers=1) as ex:
  future=ex.submit(raw,0)
  for i in range(32):
   clock=time.perf_counter();path,sim_s=future.result()
   if i<31:future=ex.submit(raw,i+1)
   with np.load(path) as z:
    frames=z['frames'];features.append(z['features'][::4].copy())
    latent,metric=request('rollout',frames)
    if i==0:
     initial,_=request('initial',frames[:1]);np.testing.assert_array_equal(initial[0],pool['z0'][i])
   path.unlink();latent=latent.astype(np.float32);latent[0]=pool['z0'][i];latents.append(latent)
   records.append({'action_idx':int(pool['action_idx'][i]),'simulate_s':sim_s,'encoder_s':metric['encode_s'],'pipeline_s':time.perf_counter()-clock})
   print(a.panel,i+1,records[-1],flush=True)
 torch.cuda.synchronize();clock=time.perf_counter()
 truth=torch.from_numpy(np.stack(latents)).cuda()
 scores=team.scorer(truth,[f[None] for f in pool['features'][:32]]).cpu().numpy()
 torch.cuda.synchronize();score_s=time.perf_counter()-clock
 from attribution_a_metrics import log_score
 from common import surface_dist
 geometry=log_score(np.stack([surface_dist(f) for f in features]),team.rule)
 np.savez(folder/'scores.npz',oracle_latent=scores,oracle_geometry=geometry,indices=pool['action_idx'][:32],labels=pool['status'][:32]>0)
 run.data.write(folder/'complete.json',{'complete':True,'task':task,'n_actions':32,'cache':'All32 observations freshly simulated and encoded; no target-cache reuse','records':records,'scorer_s':score_s,'seconds_excluding_model_load':time.perf_counter()-start,'gpu':torch.cuda.get_device_name(0),'encoder_gpus':'Two A6000 services shared by cross/within pilots','plan':run.data.sha(F/'PLAN.json')})
finally:worker.close();team.truth.close()
