import json,os,subprocess,time
from pathlib import Path
P=Path('/path/to/phyre-unc');R=P/'results/loss_design_iteration_20261019';os.chdir(P)
for index,gpu in enumerate([6,7]):
    dest=R/f'encoder{index}';dest.mkdir(parents=True,exist_ok=True)
    receipt=dest/'LAUNCH.json'
    if receipt.exists():
        old=json.loads(receipt.read_text());assert not Path('/proc',str(old['pid'])).exists(),'Existing encoder still running; do not duplicate'
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    row=next(r.split(',') for r in rows if int(r.split(',')[0])==gpu)
    assert 'A6000' in row[1] and int(row[2])<8192 and int(row[3])<20,'Insufficient memory headroom on planned shared encoder GPU'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(P/'pipeline'),XDG_CACHE_HOME='/path/to/research-assets/xdg_dirs/cache',HF_HUB_OFFLINE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    env.pop('CUBLAS_WORKSPACE_CONFIG',None);env.pop('DISABLE_ADDMM_CUDA_LT',None)
    with (dest/'server.out').open('a') as f:
        proc=subprocess.Popen(['timeout','--kill-after=30','72030',str(P/'.venv-ml/bin/python'),'-u',str(R/'code/serve_encoder.py'),'--index',str(index),'--lifetime','72000'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.write_text(json.dumps({'pid':proc.pid,'gpu':gpu,'started':time.time(),'max_seconds':72000,'occupancy_note':'Shares low-usage GPU with existing processes; no processes stopped orpreempted','memory_used_MiB_before':int(row[2])},indent=2)+'\n')
    print('ENCODER_STARTING',index,proc.pid,flush=True)
