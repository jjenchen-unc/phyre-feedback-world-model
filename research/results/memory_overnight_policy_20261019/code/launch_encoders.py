import json,os,subprocess,time
from pathlib import Path
P=Path('/path/to/phyre-unc');R=P/'results/memory_overnight_policy_20261019';os.chdir(P)
for index,gpu in enumerate([4,5]):
    dest=R/f'encoder{index}';dest.mkdir(parents=True,exist_ok=True)
    receipt=dest/'LAUNCH.json'
    if receipt.exists():
        old=json.loads(receipt.read_text());assert not Path('/proc',str(old['pid'])).exists(),'Existing encoder still running; do not duplicate'
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    row=next(r.split(',') for r in rows if int(r.split(',')[0])==gpu)
    assert 'A6000' in row[1] and int(row[2])<100,'GPU no longer free'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(P/'pipeline'),XDG_CACHE_HOME='/path/to/research-assets/xdg_dirs/cache',HF_HUB_OFFLINE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    env.pop('CUBLAS_WORKSPACE_CONFIG',None);env.pop('DISABLE_ADDMM_CUDA_LT',None)
    with (dest/'server.out').open('a') as f:
        proc=subprocess.Popen(['timeout','--kill-after=30','21630',str(P/'.venv-ml/bin/python'),'-u',str(R/'code/serve_encoder.py'),'--index',str(index),'--lifetime','21600'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.write_text(json.dumps({'pid':proc.pid,'gpu':gpu,'started':time.time(),'max_seconds':21600},indent=2)+'\n')
    print('ENCODER_STARTING',index,proc.pid,flush=True)
