"""Submit the authorized within evaluation chain once; preserve each job receipt."""
import json
from pathlib import Path
import subprocess

P=Path('/path/to/phyre-unc')
D=P/'results/support_within_20261004'
path=D/'jobs.json'
jobs=json.loads(path.read_text()) if path.exists() else {
    'selection':34405,'smoke':34406,'candidates':34407,
    'scope':'100 fresh generated scenes from 10 seen templates; evaluate same 9 trained models, no retraining'}


def save():
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(jobs,indent=2)+'\n');temp.replace(path)


def submit(name,deps,gpu,cpus,mem,time,array=None):
    if name in jobs:return jobs[name]
    cmd=['sbatch','--parsable','--partition=a6000',f'--cpus-per-task={cpus}',f'--mem={mem}G',f'--time={time}',
        '--dependency=afterok:'+':'.join(str(jobs[d]) for d in deps),f'--job-name=within_{name}',
        '--output='+str(P/'logs'/f'within_{name}_%A_%a.out')]
    if gpu:cmd+=['--gres=gpu:1']
    if array:cmd+=['--array='+array]
    cmd+=[str(D/(name+'.sbatch'))]
    result=subprocess.run(cmd,check=True,text=True,capture_output=True,cwd=P)
    job=int(result.stdout.strip().split(';')[0]);jobs[name]=job;save();print(name,job,flush=True)
    return job


jobs.setdefault('independent_audit',34417)
jobs['rollouts_dependencies']=[34408,34417]
save()
submit('prefilter',['smoke','candidates'],True,4,32,'08:00:00','0-1%2')
submit('rollouts',['prefilter','independent_audit'],False,16,32,'24:00:00')
submit('encode',['rollouts'],True,4,32,'48:00:00','0-1%2')
submit('pack',['encode'],False,8,40,'08:00:00')
submit('base',['pack'],True,4,32,'12:00:00','0-1%2')
submit('prepare',['base'],False,4,32,'01:00:00')
submit('eval',['prepare','smoke'],True,4,32,'24:00:00','0-8%2')
submit('summary',['eval'],False,4,24,'02:00:00')
