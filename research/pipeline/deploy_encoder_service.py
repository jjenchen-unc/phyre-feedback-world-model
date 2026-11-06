"""A6000 canonical trajectory encoding for A5000 training workers.

Only arrays of observations cross this file queue. No predictions, residuals,
tokens, labels, model updates or action decisions are delegated to the encoder.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import numpy as np
import deploy_pool_data as data


def request(frames):
    queue=data.R/'encoding_queue';queue.mkdir(parents=True,exist_ok=True,mode=0o700)
    key=f'{time.time_ns()}_{uuid.uuid4().hex}'
    source=queue/(key+'.npy');target=queue/(key+'.result.npy');error=queue/(key+'.error.json');done=queue/(key+'.done.json')
    data.save_npy(source,frames)
    data.write(queue/(key+'.request.json'),{'input':str(source),'output':str(target),
               'error':str(error),'done':str(done),'plan':data.sha(data.R/'PLAN.json'),'method':'rollout'})
    start=time.time()
    while not done.exists():
        if error.exists():raise RuntimeError(data.read(error))
        if time.time()-start>6*3600:raise TimeoutError('Canonical encoder request exceeded six hours')
        time.sleep(.1)
    assert data.read(done)['plan']==data.sha(data.R/'PLAN.json')
    result=np.load(target);assert result.shape==(32,196,128) and np.isfinite(result).all()
    target.unlink();source.unlink();done.unlink()
    lock=queue/(key+'.request.lock')
    if lock.exists():lock.unlink()
    return result


def serve():
    os.environ.pop('DISABLE_ADDMM_CUDA_LT',None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.device_count()==1
    assert 'A6000' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0)
    m=data.runtime().m;encoder=m.Encoder(m.PCA)
    queue=data.R/'encoding_queue';queue.mkdir(parents=True,exist_ok=True,mode=0o700)
    identity=f'{os.uname().nodename}_{os.getpid()}'
    done=0;last=0
    while not (data.R/'SUMMARY.json').exists() and not (data.R/'STOP_ENCODERS.json').exists():
        if time.time()-last>15:
            data.write(data.R/'encoder_services'/(identity+'.json'),
                {'pid':os.getpid(),'time':time.time(),'completed':done,
                 'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),
                 'job':os.environ.get('SLURM_JOB_ID'),'plan':data.sha(data.R/'PLAN.json')})
            last=time.time()
        found=False
        for path in sorted(queue.glob('*.request.json')):
            lockpath=path.with_suffix('.lock')
            with lockpath.open('w') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if not path.exists():continue
                spec=data.read(path);assert spec['plan']==data.sha(data.R/'PLAN.json')
                try:
                    assert spec['method']=='rollout'
                    frames=np.load(spec['input'])
                    latent=encoder.rollout(frames)
                    data.save_npy(Path(spec['output']),latent)
                except Exception:
                    import traceback
                    data.write(spec['error'],{'error':traceback.format_exc()})
                    raise
                path.unlink();done+=1;found=True
            # Publish completion only after removing the request and releasing
            # its lock; the client can then safely remove inputs and lock files.
            data.write(spec['done'],{'plan':spec['plan'],'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid)})
            break
        if not found:time.sleep(.1)


def launch_shared():
    # Explicitly share only our existing six-GPU oracle allocation. Other users'
    # physical GPUs6/7 are excluded. The old oracle processes remain running.
    assert os.environ.get('SLURM_JOB_ID')=='35314'
    plan=data.read(data.OLD/'PLAN.json');gpus=plan['gpu_uuids'];assert len(gpus)==6
    processes=[];logs=[]
    folder=data.R/'encoder_services';folder.mkdir(parents=True,exist_ok=True)
    try:
        for i,gpu in enumerate(gpus):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            log=(folder/f'shared_{i}.out').open('a');logs.append(log)
            processes.append(subprocess.Popen([str(data.P/'.venv-ml/bin/python'),'-u',__file__,'serve'],env=env,stdout=log,stderr=subprocess.STDOUT))
        while processes:
            for process in list(processes):
                code=process.poll()
                if code is not None:
                    if code:raise RuntimeError(f'Encoder service failed: {code}')
                    processes.remove(process)
            time.sleep(5)
    finally:
        for process in processes:process.terminate()
        for process in processes:process.wait(timeout=30)
        for log in logs:log.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['serve','shared']);a=ap.parse_args()
    if a.mode=='shared':launch_shared()
    else:serve()
