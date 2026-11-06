import datetime,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/path/to/phyre-unc/results/query_token_staged_20261005')
sys.path.insert(0,str(ROOT/'code'))
import query_token_staged as s
s.checked(ROOT)
record=ROOT/'recovery_20261006/RECEIPT.json'
assert not record.exists(),'Recovery already submitted; inspect receipt, do not duplicate'
jobs=s.f.read(ROOT/'jobs.json');assert jobs['q0_eval']==34544
assert s.f.read(ROOT/'TRAINING_AUDIT.json')['passed']
cmd=['sbatch','--parsable','--partition=a6000','--nodelist=oprime.ib','--cpus-per-task=4','--mem=32G',
     '--time=24:00:00','--gres=gpu:1','--array=0-5%5','--job-name=qts_q0_eval_retry',
     '--output=/path/to/phyre-unc/logs/qts_q0_eval_retry_%A_%a.out',str(ROOT/'recovery_20261006/q0_eval.sbatch')]
job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0])
s.f.write_json(record,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'failed_q0_eval':34544,'replacement_q0_eval':job,'cause':'static_case wrote scores.npz before mkdir',
    'fix':'Precreate the100static task directories in each evaluation job; no numerical code/manifest/checkpoint changes',
    'training_audit_sha256':s.f.digest(ROOT/'TRAINING_AUDIT.json'),
    'wrapper_sha256':s.f.digest(ROOT/'recovery_20261006/prepare_output_dirs.py')})
jobs['failed_q0_eval_20261006']=jobs['q0_eval'];jobs['q0_eval']=job;s.f.write_json(ROOT/'jobs.json',jobs)
subprocess.run(['scontrol','update',f'JobId={jobs["updater_eval"]}',f'Dependency=afterok:{job}'],check=True)
print('RECOVERED_Q0_EVAL',job,'UPDATER_DEPENDENCY_UPDATED',jobs['updater_eval'],flush=True)
