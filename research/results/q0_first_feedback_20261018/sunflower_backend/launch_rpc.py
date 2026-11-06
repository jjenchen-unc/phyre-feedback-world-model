"""Start a bounded validation encoder only beside our own oracle process."""
from pathlib import Path
import argparse,json,os,subprocess,time
B=Path(__file__).resolve().parent
P=B.parents[2]
os.chdir(P)
folder=B/'rpc';folder.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--restart',action='store_true');ap.add_argument('--lifetime',type=int,default=86400)
args=ap.parse_args();port=0
if (folder/'ENDPOINT.json').exists():
    assert args.restart,'Existing endpoint: use --restart only after its server has exited'
    prior=json.loads((folder/'ENDPOINT.json').read_text())
    assert not Path('/proc',str(prior['pid'])).exists(),'Prior encoder process is still alive'
    port=prior['port']
if (folder/'STOP').exists():
    assert args.restart
    (folder/'STOP').unlink()
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
row=next(r.split(',') for r in rows if int(r.split(',')[0])==4)
gpu=row[1].strip();assert int(row[2])<12000
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
pids=[r.split(',')[1].strip() for r in apps if r.split(',')[0].strip()==gpu]
for pid in pids:
    assert subprocess.check_output(['ps','-o','user=','-p',pid],text=True).strip()=='jjenchen'
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='4',PYTHONPATH=str(P/'pipeline'),XDG_CACHE_HOME='/path/to/research-assets/xdg_dirs/cache',
           HF_HUB_OFFLINE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
env.pop('CUBLAS_WORKSPACE_CONFIG',None);env.pop('DISABLE_ADDMM_CUDA_LT',None)
started=time.time()
with (folder/'server.out').open('a') as f:
    proc=subprocess.Popen(['timeout','--kill-after=60',str(args.lifetime+60),str(P/'.venv-ml/bin/python'),'-u',str(B/'encoder_rpc.py'),'--port',str(port),'--lifetime',str(args.lifetime)],
                          env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(folder/'LAUNCH.json').write_text(json.dumps({'pid':proc.pid,'gpu':4,'existing_own_oracle_pids':pids,'time':started,'bounded_hours':args.lifetime/3600},indent=2))
print('ENCODER_STARTING',proc.pid,flush=True)
while time.time()-started<180:
    if proc.poll() is not None:raise RuntimeError('Encoder exited during startup; inspect rpc/server.out')
    endpoint=folder/'ENDPOINT.json'
    if endpoint.exists() and endpoint.stat().st_mtime>=started:
        print('ENCODER_READY',proc.pid,flush=True);break
    time.sleep(1)
else:raise TimeoutError('Encoder startup exceeded 180 seconds; inspect process before retrying')
