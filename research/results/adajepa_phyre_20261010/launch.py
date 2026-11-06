"""Attach AdaJEPA lanes to freed GPU slots; preserve the total four-GPU ceiling."""
import json,subprocess
from pathlib import Path
R=Path('/path/to/phyre-unc/results/adajepa_phyre_20261010');P=R.parents[1];M=P/'results/small_official_20261010'
def submit(name,args,deps=(),gpu=True,hours=48,ada=False):
 jobs=json.loads((R/'jobs.json').read_text()) if (R/'jobs.json').exists() else {'encoding_check':34741,'ada_smoke':34742}
 if name in jobs:return jobs[name]
 cmd=['sbatch','--parsable','--partition=a6000','--nodelist=oprime.ib',
  '--cpus-per-task='+('12' if name=='labels_fixed' else '4'),'--mem=48G',f'--time={hours}:00:00',
  '--job-name=ada_'+name,'--output='+str(P/'logs'/('ada_'+name+'_%j.out'))]
 if gpu:cmd+=['--gres=gpu:1']
 if deps:cmd+=['--dependency=afterok:'+':'.join(map(str,deps))]
 cmd+=[str(R/'run.sbatch' if ada else M/'recovery_20261010/run.sbatch')]+args
 job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0]);jobs[name]=job
 temp=R/'jobs.tmp';temp.write_text(json.dumps(jobs,indent=2));temp.replace(R/'jobs.json');print(name,job,flush=True);return job
if __name__=='__main__':
 assert (R/'PLAN.json').exists() and json.loads((M/'ENCODING_CHECK_FIXED.json').read_text())['passed']
 shards=[submit(f'shortlist_fixed_{s}',['shortlist','--shard',str(s)],[34741,34742],hours=12) for s in range(4)]
 labels=submit('labels_fixed',['labels'],shards,gpu=False,hours=12)
 smoke=submit('eval_smoke_fixed',['smoke_eval'],[labels],hours=2)
 cross=submit('memory_cross',['lane','--panel','cross'],[smoke])
 within=[submit(f'memory_within_s{s}',['lane','--panel','within','--seed',str(s)],[smoke]) for s in range(3)]
 memory_summary=submit('memory_summary',['summary'],[cross]+within,gpu=False,hours=2)
 a=submit('tta_cross',['lane','--panel','cross'],[cross],ada=True)
 b=submit('tta_within',['lane','--panel','within'],[within[0]],ada=True)
 submit('tta_summary',['summary'],[a,b,memory_summary],gpu=False,hours=2,ada=True)
