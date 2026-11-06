from transfer_core import *
import subprocess
checked();assert not (R/'JOBS.json').exists();jobs={};lj=read(L/'JOBS.json')
def submit(name,args,dep=None):
 cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes']
 if dep:cmd+=['--dependency='+dep]
 cmd+=list(map(str,args));j=subprocess.check_output(cmd,cwd=P,text=True).strip().split(';')[0];assert j.isdigit();jobs[name]=j;write(R/'JOBS.json',jobs);print(name,j,flush=True);return j
s=submit('smoke',[R/'smoke.sbatch'],'afterok:'+lj['diagnostic'])
f=submit('first',[R/'first.sbatch'],'afterok:'+s)
h=submit('history',[R/'history.sbatch'],'afterok:'+s)
d=submit('decision',[R/'cpu.sbatch',R/'code/transfer_summary.py','decision'],'afterok:'+f+':'+h)
e=submit('policies',[R/'policies.sbatch'],'afterok:'+d)
r=submit('final',[R/'cpu.sbatch',R/'code/transfer_summary.py','final'],'afterok:'+e)
submit('joint',[R/'cpu.sbatch',R/'code/transfer_joint.py'],'afterok:'+r+':'+lj['final_analysis'])
# Shared ownedencoder services mustoutlive both experiments. Existing job stayssame.
wait_jobs=[v for k,v in lj.items() if k!='encoder_cleanup']+[s,f,h,d,e,r]
subprocess.check_call(['scontrol','update','JobId='+lj['encoder_cleanup'],'Dependency=afterany:'+':'.join(wait_jobs)])
write(R/'SHARED_ENCODER_CLEANUP_DEPENDENCY.json',{'job':lj['encoder_cleanup'],'waits_for':wait_jobs,'scope':'Dependency only; existing loss jobs/code unchanged'})
