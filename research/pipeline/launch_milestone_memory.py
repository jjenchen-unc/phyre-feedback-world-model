"""One Slurm allocation, six explicit GPU lanes, resumable milestone DAG."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT=Path('/path/to/phyre-unc')
ROOT=PROJECT/'results/milestone_memory_20261015'
PYTHON=PROJECT/'.venv-ml/bin/python'
RUNNER=ROOT/'code/milestone_memory.py'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run inside the six-GPU allocation'
    plan=json.loads((ROOT/'PLAN.json').read_text());gpus=plan['gpu_uuids'];assert len(gpus)==6
    # Explicit UUIDs avoid CUDA's PCI-order/index ambiguity on this shared host.
    allocated=os.environ.get('SLURM_JOB_GPUS','').split(',')
    assert sorted(allocated)==[str(i) for i in range(6)], ('Unexpected GPU allocation',allocated)
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.split(',')[0].strip() in gpus for line in busy.splitlines()), busy
    attempts=ROOT/'attempts'/f'job_{os.environ["SLURM_JOB_ID"]}'
    attempts.mkdir(parents=True,exist_ok=False)
    pending=[];done=set();failed={};active={};started=time.time()
    def add(name,args,deps=()):pending.append({'name':name,'args':args,'deps':list(deps)})
    pre=['preflight_cross','preflight_within']
    for panel in ['cross','within']:add('preflight_'+panel,['preflight','--panel',panel])
    for panel in ['cross','within']:
        for kind in ['residual','zero']:
            add(f'train_{panel}_{kind}',['train','--panel',panel,'--kind',kind],pre)
    # Training enters first; remaining lanes begin evaluation immediately.
    for kind in ['tta','base','q0','residual','zero']:
        for panel in ['cross','within']:
            for shard in range(4):
                deps=pre+([f'train_{panel}_{kind}'] if kind in ['residual','zero'] else [])
                add(f'eval_{panel}_{kind}_{shard}', ['eval','--panel',panel,'--kind',kind,'--shard',str(shard)],deps)
    allnames=[task['name'] for task in pending]
    add('summary',['summary'],allnames)
    def stop(signum,frame):
        for info in active.values():os.killpg(info['process'].pid,signal.SIGTERM)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    lastsample=0
    while pending or active:
        for lane,info in list(active.items()):
            code=info['process'].poll()
            if code is None:continue
            info['log'].close();record={k:v for k,v in info.items() if k not in ['process','log']}
            record.update(returncode=code,finished=time.time(),seconds=time.time()-info['started'])
            write(attempts/(info['name']+'.json'),record)
            if code==0:done.add(info['name'])
            else:failed[info['name']]=code
            print('FINISHED',info['name'],'code',code,flush=True);del active[lane]
        for task in list(pending):
            if any(dep in failed for dep in task['deps']):
                failed[task['name']]='dependency_failed';pending.remove(task)
        for lane,gpu in enumerate(gpus):
            if lane in active:continue
            task=next((t for t in pending if all(d in done for d in t['deps'])),None)
            if task is None:continue
            pending.remove(task);name=task['name'];log=(attempts/(name+'.out')).open('w')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                PYTHONNOUSERSITE='1',PYTHONPATH=str(ROOT/'code')+':'+str(PROJECT/'pipeline'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
            command=[str(PYTHON),'-u',str(RUNNER)]+task['args']
            process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[lane]={'name':name,'pid':process.pid,'gpu_uuid':gpu,'lane':lane,'started':time.time(),
                'command':command,'process':process,'log':log}
            print('STARTED',name,'gpu',lane,'pid',process.pid,flush=True)
        status={'job':os.environ['SLURM_JOB_ID'],'started':started,'updated':time.time(),
            'done':sorted(done),'failed':failed,'pending':[t['name'] for t in pending],
            'active':[{k:v for k,v in i.items() if k not in ['process','log']} for i in active.values()]}
        write(ROOT/'STATUS.json',status)
        if time.time()-lastsample>=60:
            sample={'time':time.time(),'active':status['active']}
            for key,query in [('gpus','--query-gpu=index,uuid,memory.used,utilization.gpu'),('processes','--query-compute-apps=gpu_uuid,pid,used_memory')]:
                sample[key]=subprocess.check_output(['nvidia-smi',query,'--format=csv,noheader'],text=True).strip()
            with (attempts/'telemetry.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
            lastsample=time.time()
        if pending or active:time.sleep(5)
    write(attempts/'complete.json',dict(status,complete=not failed))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
