"""Submit dependencies; at most four GPU jobs can be active in this DAG."""
import json,subprocess
from pathlib import Path
R=Path('/path/to/phyre-unc/results/small_official_20261010');P=R.parents[1]
def submit(name,args,deps=(),gpu=True,hours=48,array=None):
 jobs=json.loads((R/'jobs.json').read_text())
 if name in jobs:return jobs[name]
 cmd=['sbatch','--parsable','--partition=a6000','--nodelist=oprime.ib',
  '--cpus-per-task='+('12' if name=='labels' else '4'),'--mem=48G',f'--time={hours}:00:00',
  '--job-name=smoff_'+name,'--output='+str(P/'logs'/('smoff_'+name+'_%j.out'))]
 if gpu:cmd+=['--gres=gpu:1']
 if deps:cmd+=['--dependency=afterok:'+':'.join(map(str,deps))]
 cmd+=[str(R/'run.sbatch')]+args
 job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0]);jobs[name]=job
 temp=R/'jobs.tmp';temp.write_text(json.dumps(jobs,indent=2));temp.replace(R/'jobs.json')
 print(name,job,flush=True);return job
if __name__=='__main__':
 assert (R/'PLAN.json').exists()
 jobs=json.loads((R/'jobs.json').read_text())
 smoke=submit('smoke_train',['smoke_train'],[jobs['encoding_check']],hours=4)
 shards=[submit(f'shortlist_{s}',['shortlist','--shard',str(s)],[smoke,jobs['candidates']],hours=12) for s in range(4)]
 labels=submit('labels',['cpu_labels'],shards,gpu=False,hours=12)
 evalsmoke=submit('smoke_eval',['smoke_eval'],[labels],hours=2)
 lanes=[submit('cross',['lane','--panel','cross'],[evalsmoke],hours=48)]
 lanes += [submit(f'within_s{s}',['lane','--panel','within','--seed',str(s)],[evalsmoke],hours=48) for s in range(3)]
 submit('summary',['summary'],lanes,gpu=False,hours=2)
