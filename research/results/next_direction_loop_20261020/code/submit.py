from direction_runtime import *
import subprocess

checked();assert not (R/'JOBS.json').exists();jobs={}
def submit(name,args,dependency=None):
    cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--job-name=direction_'+name,'--output='+str(R/(name+'_%A_%a.out'))]
    if dependency:cmd+=['--dependency=afterok:'+dependency]
    cmd+=list(map(str,args));j=subprocess.check_output(cmd,cwd=P,text=True).strip().split(';')[0];assert j.isdigit();jobs[name]=j;write(R/'JOBS.json',jobs);print(name,j,flush=True);return j
cpu,gpu,code=R/'cpu.sbatch',R/'gpu.sbatch',R/'code'
r1=submit('round1',[cpu,code/'round1.py'])
st=submit('smoke_train',['--array=0-7',gpu,'smoke_train'],r1)
sa=submit('smoke_audit',[cpu,code/'train_head.py','--audit','--smoke'],st)
se=submit('smoke_eval',[gpu,'smoke_eval'],sa)
tr=submit('train',['--array=0-7',gpu,'train'],se)
ta=submit('train_audit',[cpu,code/'train_head.py','--audit'],tr)
ev=submit('discovery',['--array=0-7',gpu,'discovery'],ta)
de=submit('decision',[cpu,code/'decide.py','discovery'],ev)
co=submit('confirmation',['--array=0-7',gpu,'confirmation'],de)
submit('final',[cpu,code/'decide.py','final'],co)
