"""Submit the authorized three-seed staged experiment with explicit dependencies."""
import json
import subprocess
from pathlib import Path
R=Path('/path/to/phyre-unc/results/prefix_staged_20261008')
P=R.parent.parent

def write(x):
    p=R/'jobs.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)

def submit(name,args,dependencies=(),gpu=True,hours=48):
    jobs=json.loads((R/'jobs.json').read_text()) if (R/'jobs.json').exists() else {}
    if name in jobs:return jobs[name]
    cmd=['sbatch','--parsable','--partition=a6000','--nodelist=oprime.ib',
        '--cpus-per-task='+('4' if gpu else '2'),'--mem='+('48G' if gpu else '8G'),
        f'--time={hours}:00:00','--job-name=pxs_'+name,'--output='+str(P/'logs'/('pxs_'+name+'_%j.out'))]
    if gpu:cmd+=['--gres=gpu:1']
    if dependencies:cmd+=['--dependency=afterok:'+':'.join(map(str,dependencies))]
    cmd+=[str(R/'run.sbatch')]+args
    job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0]);jobs[name]=job;write(jobs)
    print(name,job,flush=True);return job

def main():
    m=json.loads((R/'manifest.json').read_text());assert m['stage1']['updates']==52272
    assert json.loads((R/'DATA_AUDIT.json').read_text())['passed']
    smoke=submit('smoke',['smoke'],hours=4)
    last=[]
    for seed in m['seeds']:
        q=submit(f'q0_s{seed}',['train','--stage','q0','--seed',str(seed)],[smoke])
        real=submit(f'residual_s{seed}',['train','--stage','residual','--seed',str(seed)],[q])
        zero=submit(f'zero_s{seed}',['train','--stage','zero','--seed',str(seed)],[q])
        for split in ['cross','within']:
            qe=submit(f'q0_s{seed}_{split}',['eval','--stage','q0','--seed',str(seed),'--split',split],[q],hours=24)
            last.append(qe)
            for stage,job in [('residual',real),('zero',zero)]:
                last.append(submit(f'{stage}_s{seed}_{split}',['eval','--stage',stage,'--seed',str(seed),'--split',split],[job,qe],hours=24))
    submit('summary',['summary'],last,gpu=False,hours=2)
if __name__=='__main__':main()
