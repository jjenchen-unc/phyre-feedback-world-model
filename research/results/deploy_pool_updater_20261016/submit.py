"""Submit the complete gated Sunflower DAG; resume submission without duplication."""
import json
from pathlib import Path
import subprocess

ROOT=Path('/path/to/phyre-unc/results/deploy_pool_updater_20261016')
path=ROOT/'JOBS.json'
jobs=json.loads(path.read_text()) if path.exists() else {'pool_validation':'35692','ten_gpu_rankcheck':'35690'}


def submit(key,script,args=(),deps=(),options=()):
    if key in jobs:return jobs[key]
    command=['sbatch','--parsable']+list(options)
    if deps:command+=['--dependency=afterok:'+':'.join(map(str,deps))]
    command+=[str(ROOT/script)]+list(args)
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    assert job.isdigit(),job
    jobs[key]=job
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(jobs,indent=2)+'\n');tmp.replace(path)
    print(key,job,flush=True);return job


smoke={panel:submit('smoke_pools_'+panel,'single.sbatch',['smoke_pools','--panel',panel],
                   [jobs['pool_validation']]) for panel in ['cross','within']}
collection=submit('collection','data.sbatch',['collect'],[jobs['pool_validation']],['--array=0-39%30'])
preflights=[]
for panel in ['cross','within']:
    for kind in ['residual','zero']:
        preflights.append(submit('preflight_'+panel+'_'+kind,'team.sbatch',
                          ['preflight','--panel',panel,'--kind',kind],[smoke[panel]],['--time=02:00:00']))
# Lift the temporary cap once the ten-GPU numerical/gradient checks release their lane.
throttle=ROOT/'throttle.sbatch'
throttle.write_text('#!/bin/bash\n#SBATCH --partition=defq\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=1\n#SBATCH --time=00:05:00\n'
                    'set -euo pipefail\nscontrol update JobId='+collection+' ArrayTaskThrottle=40\n')
submit('lift_collection_throttle','throttle.sbatch',deps=preflights)
audit=submit('data_audit','single.sbatch',['audit_data'],[collection])
evaluations=[]
for panel in ['cross','within']:
    for kind in ['residual','zero']:
        trained=submit('train_'+panel+'_'+kind,'team.sbatch',
                       ['train','--panel',panel,'--kind',kind],preflights+[audit])
        for shard in range(4):
            evaluations.append(submit(f'eval_{panel}_{kind}_{shard}','single.sbatch',
                    ['eval','--panel',panel,'--kind',kind,'--shard',str(shard)],[trained]))
submit('summary','single.sbatch',['summary'],evaluations)
