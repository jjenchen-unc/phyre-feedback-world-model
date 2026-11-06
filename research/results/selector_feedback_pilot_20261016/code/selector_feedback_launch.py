"""Two cluster controllers and a bounded six-GPU training/evaluation dispatcher."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P=Path('/path/to/phyre-unc')
R=P/'results/selector_feedback_pilot_20261016'
ML=str(P/'.venv-ml/bin/python')


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def wait_file(path,timeout=48*3600):
    start=time.time()
    while not path.exists():
        if (R/'FAILED.json').exists():raise RuntimeError((R/'FAILED.json').read_text())
        if time.time()-start>timeout:raise TimeoutError(str(path))
        time.sleep(30)
    return json.loads(path.read_text())


def submit(script,args=(),dependency=None,array=None):
    cmd=['sbatch','--parsable']
    if dependency:cmd+=['--dependency=afterok:'+str(dependency)]
    if array:cmd+=['--array='+array]
    return subprocess.check_output(cmd+[str(R/script),*args],text=True).strip()


def sunflower():
    # Scheduler dependency ensures current full100k comparison is finished first.
    subprocess.run([str(P/'runtime/sunflower_phyre_python'),str(R/'code/selector_feedback_data.py'),'prepare'],check=True)
    wait_file(R/'ENCODING_COMPLETE.json')
    jobs={}
    jobs['collection']=submit('data.sbatch',array='0-39')
    jobs['audit']=submit('audit.sbatch',dependency=jobs['collection'])
    write(R/'JOBS_SUNFLOWER.json',jobs)
    print('SUBMITTED',jobs,flush=True)


def oprime():
    wait_file(R/'PLAN.json')
    jobs={'encoding':submit('encode.sbatch')};write(R/'JOBS_OPRIME.json',jobs)
    wait_file(R/'ENCODING_COMPLETE.json')
    audit=wait_file(R/'DATA_AUDIT.json');assert audit['passed']
    jobs['pilot']=submit('pilot.sbatch');write(R/'JOBS_OPRIME.json',jobs)
    print('SUBMITTED',jobs,flush=True)


def gpu_inventory():
    raw=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    assert len(raw)==6 and set(raw)==set(map(str,range(6))),raw
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
    mapping={a.strip():b.strip() for a,b in (line.split(',') for line in lines)}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(mapping[i] in apps for i in raw),'Allocated GPU already has a compute process'
    return [mapping[i] for i in raw]


def dispatch(jobs,gpus):
    pending=list(jobs);active={};completed=set();logs=[]
    try:
        while pending or active:
            for gpu,(name,proc,log) in list(active.items()):
                status=proc.poll()
                if status is None:continue
                log.close();del active[gpu]
                if status!=0:raise RuntimeError(f'{name} exited {status}')
                completed.add(name)
            for gpu in gpus:
                if gpu in active:continue
                eligible=next((j for j in pending if set(j.get('after',[]))<=completed),None)
                if eligible is None:continue
                pending.remove(eligible);name=eligible['name']
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
                log=(R/'logs'/(name+'.out')).open('a');logs.append(log)
                cmd=[ML,'-u',str(R/'code/selector_feedback_run.py'),*eligible['args']]
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=P)
                active[gpu]=(name,proc,log)
            write(R/'STATUS.json',{'time':time.time(),'active':{g:v[0] for g,v in active.items()},
                                  'completed':sorted(completed),'pending':[j['name'] for j in pending]})
            if pending and not active:raise RuntimeError('Unresolvable pilot dependency graph')
            time.sleep(5)
    finally:
        for _,proc,log in active.values():
            if proc.poll() is None:proc.terminate()
        for _,proc,log in active.values():
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()


def pilot():
    gpus=gpu_inventory();(R/'logs').mkdir(exist_ok=True)
    checks=[]
    for panel in ['cross','within']:
        for kind in ['residual','zero']:
            checks.append({'name':f'check_{panel}_{kind}','args':['preflight','--panel',panel,'--kind',kind]})
        for kind in ['q0','tta']:
            checks.append({'name':f'check_eval_{panel}_{kind}',
                'args':['eval','--panel',panel,'--kind',kind,'--shard','0','--nshards','100']})
    dispatch(checks,gpus)
    write(R/'ALL_PREFLIGHTS_PASSED.json',{'passed':True,'time':time.time(),'checks':[j['name'] for j in checks]})
    jobs=[]
    for panel in ['cross','within']:
        for kind in ['residual','zero']:
            name=f'train_{panel}_{kind}'
            jobs.append({'name':name,'args':['train','--panel',panel,'--kind',kind]})
    for panel in ['cross','within']:
        for kind in ['q0','tta','residual','zero']:
            for shard in range(4):
                jobs.append({'name':f'eval_{panel}_{kind}_{shard}',
                    'args':['eval','--panel',panel,'--kind',kind,'--shard',str(shard),'--nshards','4'],
                    'after':[f'train_{panel}_{kind}'] if kind in ['residual','zero'] else []})
    dispatch(jobs,gpus)
    subprocess.run([ML,str(R/'code/selector_feedback_run.py'),'summary'],check=True,cwd=P)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['sunflower','oprime','pilot']);a=ap.parse_args()
    try:globals()[a.mode]()
    except Exception:
        import traceback
        write(R/'FAILED.json',{'phase':a.mode,'error':traceback.format_exc(),'time':time.time()})
        raise
