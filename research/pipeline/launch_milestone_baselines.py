"""Schedule scene baselines and true-latent oracle in one six-GPU allocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

P=Path('/path/to/phyre-unc');R=P/'results/milestone_baselines_20261015'
SCENE=P/'results/milestone_scene_baseline_20261015';ORACLE=P/'results/milestone_oracle_latent_20261015'
PY=P/'.venv-ml/bin/python'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    assert os.environ.get('SLURM_JOB_ID')
    assert sorted(os.environ.get('SLURM_JOB_GPUS','').split(','))==[str(i) for i in range(6)]
    for path in [SCENE/'PLAN.json',SCENE/'INPUT_AUDIT.json',ORACLE/'PLAN.json']:assert path.exists(),path
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
    gpus=[line.split(',')[1].strip() for line in lines if int(line.split(',')[0])<6];assert len(gpus)==6
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.split(',')[0].strip() in gpus for line in busy.splitlines()),busy
    folder=R/'attempts'/('job_'+os.environ['SLURM_JOB_ID']);folder.mkdir(parents=True,exist_ok=False)
    tasks=[];active={};done=set();failed={};start=time.time()
    def add(name,root,args,deps=()):tasks.append({'name':name,'root':str(root),'args':args,'deps':list(deps)})
    for panel in ['cross','within']:add('scene_'+panel,SCENE,['eval','--panel',panel])
    for panel in ['cross','within']:add('oracle_smoke_'+panel,ORACLE,['smoke','--panel',panel])
    add('scene_summary',SCENE,['summary'],['scene_cross','scene_within'])
    smoke=['oracle_smoke_cross','oracle_smoke_within'];full=[]
    for panel in ['cross','within']:
        for shard in range(3):
            name=f'oracle_{panel}_{shard}';full.append(name)
            add(name,ORACLE,['eval','--panel',panel,'--shard',str(shard),'--nshards','3'],smoke)
    add('oracle_summary',ORACLE,['summary'],full)
    def stop(signum,frame):
        for info in active.values():os.killpg(info['process'].pid,signal.SIGTERM)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while tasks or active:
        for lane,info in list(active.items()):
            code=info['process'].poll()
            if code is None:continue
            info['log'].close();record={k:v for k,v in info.items() if k not in ['process','log']}
            write(folder/(info['name']+'.json'),dict(record,returncode=code,finished=time.time()))
            if code==0:done.add(info['name'])
            else:failed[info['name']]=code
            print('FINISHED',info['name'],code,flush=True);del active[lane]
        for task in list(tasks):
            if any(dep in failed for dep in task['deps']):failed[task['name']]='dependency_failed';tasks.remove(task)
        for lane,gpu in enumerate(gpus):
            if lane in active:continue
            task=next((t for t in tasks if all(dep in done for dep in t['deps'])),None)
            if task is None:continue
            tasks.remove(task);root=Path(task['root']);script='milestone_scene_baseline.py' if root==SCENE else 'milestone_oracle_latent.py'
            command=[str(PY),'-u',str(root/'code'/script)]+task['args'];log=(folder/(task['name']+'.out')).open('w')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',PYTHONPATH=str(root/'code'))
            process=subprocess.Popen(command,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[lane]={'name':task['name'],'gpu_uuid':gpu,'pid':process.pid,'started':time.time(),'command':command,'process':process,'log':log}
            print('STARTED',task['name'],'gpu',lane,flush=True)
        status={'job':os.environ['SLURM_JOB_ID'],'started':start,'updated':time.time(),'done':sorted(done),'failed':failed,
            'pending':[t['name'] for t in tasks],'active':[{k:v for k,v in a.items() if k not in ['process','log']} for a in active.values()]}
        write(R/'STATUS.json',status)
        if tasks or active:time.sleep(5)
    write(folder/'complete.json',dict(status,complete=not failed))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
