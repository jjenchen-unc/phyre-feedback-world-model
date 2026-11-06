"""Submit smoke-gated full ABCDE evaluation, preserving old jobs and artifacts."""
import json, subprocess, hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
assert json.loads((R/'CPU_AUDIT.json').read_text())['passed']
m=json.loads((R/'manifest.json').read_text())
assert hashlib.sha256((R/'evaluate_full.py').read_bytes()).hexdigest()==m['wrapper_sha256']
jobs=json.loads((R/'jobs.json').read_text()) if (R/'jobs.json').exists() else {}
def submit(name,dependency=None,array=None,gpu=True):
    if name in jobs:return jobs[name]
    args=['sbatch','--parsable','--partition=a6000','--nodelist=oprime.ib',
          '--cpus-per-task='+('4' if gpu else '2'),'--mem='+('32G' if gpu else '8G'),
          '--time='+('24:00:00' if name=='eval' else '02:00:00'),
          '--job-name=qtd_full_'+name,'--output=/path/to/phyre-unc/logs/qtd_full_'+name+'_%A_%a.out']
    if gpu:args+=['--gres=gpu:1']
    if dependency:args+=['--dependency=afterok:'+str(dependency)]
    if array:args+=['--array='+array]
    args+=[str(R/'run.sbatch'),name]
    job=int(subprocess.check_output(args,text=True).strip().split(';')[0]);jobs[name]=job
    p=R/'jobs.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(jobs,indent=2)+'\n');tmp.replace(p)
    print(name,job,flush=True);return job
smoke=submit('smoke')
evaluation=submit('eval',dependency=smoke,array='0-17%5')
submit('summary',dependency=evaluation,gpu=False)
