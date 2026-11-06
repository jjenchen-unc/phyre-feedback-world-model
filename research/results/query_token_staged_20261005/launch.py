"""Priority staged experiment DAG. GPU smoke gates all nine formal training tasks."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/path/to/phyre-unc/results/query_token_staged_20261005')
sys.path.insert(0,str(ROOT/'code'))
import query_token_staged as s


def submit(name, script, array=None, dependency=None, gpu=True, hours=24):
    jobs=s.f.read(ROOT/'jobs.json') if (ROOT/'jobs.json').exists() else {}
    if name in jobs:return jobs[name]
    args=['sbatch','--parsable','--partition=a6000','--nodelist=oprime.ib',
          '--cpus-per-task=4' if gpu else '--cpus-per-task=2',
          '--mem=32G' if gpu else '--mem=8G',f'--time={hours}:00:00',
          '--job-name=qts_'+name,'--output='+str(s.f.PROJECT/'logs'/('qts_'+name+'_%A_%a.out'))]
    if gpu:args.append('--gres=gpu:1')
    if array:args.append('--array='+array)
    if dependency:args.append('--dependency='+dependency)
    args.append(str(ROOT/script))
    job=int(subprocess.check_output(args,text=True).strip().split(';')[0])
    jobs[name]=job;s.f.write_json(ROOT/'jobs.json',jobs)
    print(name,job,flush=True);return job


def main():
    s.checked(ROOT)
    assert s.f.read(ROOT/'CPU_AUDIT.json')['passed']
    if sys.argv[1]=='bootstrap':
        smoke=submit('smoke','smoke.sbatch',hours=4)
        submit('launch_after_smoke','launch_after_smoke.sbatch',dependency=f'afterok:{smoke}',gpu=False,hours=1)
        return
    assert sys.argv[1]=='formal'
    smoke=s.f.read(ROOT/'SMOKE.json')
    assert smoke['passed'] and smoke['provenance']==s.f.provenance(ROOT)
    # Yield the second old GPU only after new smoke passes; completed scene artifacts are resumable.
    state=subprocess.check_output(['squeue','-h','-j','34414_7','-o','%T'],text=True).strip()
    if state=='RUNNING':
        subprocess.run(['scontrol','requeuehold','34414_7'],check=True)
        receipt=s.f.read(ROOT/'PRIORITY_ACTIONS.json')
        if '34414_7' not in receipt['held_pending_jobs']:receipt['held_pending_jobs'].append('34414_7')
        receipt['left_running']=[]
        receipt.setdefault('requeued_running_jobs',[]).append({'job':'34414_7','reason':'GPU smoke passed; prioritize full stage1 training; old per-scene outputs retained'})
        receipt['release_command']='scontrol release '+' '.join(receipt['held_pending_jobs'])
        s.f.write_json(ROOT/'PRIORITY_ACTIONS.json',receipt)
    stage1=submit('q0_train','q0_train.sbatch','0-2%3')
    # aftercorr lets each seed's updater start after that seed's Q0 training.
    # At most 3 Q0 jobs, and 2 per updater branch: dependencies imply <=5 total.
    real=submit('residual_train','residual_train.sbatch','0-2%2',f'aftercorr:{stage1}',hours=12)
    zero=submit('zero_train','zero_train.sbatch','0-2%2',f'aftercorr:{stage1}',hours=12)
    audit=submit('training_audit','training_audit.sbatch',dependency=f'afterok:{real}:{zero}',gpu=False,hours=1)
    static=submit('q0_eval','q0_eval.sbatch','0-5%5',f'afterok:{audit}')
    evaluation=submit('updater_eval','updater_eval.sbatch','0-11%5',f'afterok:{static}')
    cross=submit('cross_summary','cross_summary.sbatch',dependency=f'afterok:{evaluation}',gpu=False,hours=4)
    within=submit('within_summary','within_summary.sbatch',dependency=f'afterok:{evaluation}',gpu=False,hours=4)
    submit('release_old','release_old.sbatch',dependency=f'afterok:{cross}:{within}',gpu=False,hours=1)


if __name__=='__main__':main()
