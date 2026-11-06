"""Continue canonical encoding and all oracle shards after the original job ends."""
import json
import os
from pathlib import Path
import subprocess
import time

P=Path('/path/to/phyre-unc');R=P/'results/deploy_pool_updater_20261016'
PY=P/'.venv-ml/bin/python';ORACLE=P/'results/milestone_oracle_latent_20261015/recovery_20261016/recover.py'


def main():
    plan=json.loads((P/'results/milestone_memory_20261015/PLAN.json').read_text());gpus=plan['gpu_uuids']
    assert sorted(os.environ['SLURM_JOB_GPUS'].split(','))==list(map(str,range(6)))
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.split(',')[0].strip() in gpus for line in busy.splitlines()),busy
    folder=R/'continuation'/os.environ['SLURM_JOB_ID'];folder.mkdir(parents=True,exist_ok=True)
    processes={};logs=[];done={};summary=False
    def start(name,command,gpu):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        log=(folder/(name+'.out')).open('w');logs.append(log)
        processes[name]=subprocess.Popen([str(PY),'-u']+list(map(str,command)),env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        for i,gpu in enumerate(gpus):
            start(f'encoder_{i}',[R/'code/deploy_encoder_service.py','serve'],gpu)
            panel='cross' if i<3 else 'within';shard=i%3
            start(f'oracle_{panel}_{shard}',[ORACLE,'eval','--panel',panel,'--shard',str(shard)],gpu)
        while processes:
            for name,process in list(processes.items()):
                code=process.poll()
                if code is None:continue
                done[name]=code;del processes[name]
                if code:raise RuntimeError((name,code))
            if not summary and all(name in done for name in [f'oracle_{p}_{s}' for p in ['cross','within'] for s in range(3)]):
                subprocess.run([str(PY),str(ORACLE),'summary'],check=True);summary=True
            temp=folder/'status.tmp';temp.write_text(json.dumps({'active':list(processes),'done':done,'oracle_summary':summary,'time':time.time()},indent=2));temp.replace(folder/'STATUS.json')
            time.sleep(5)
    finally:
        for process in processes.values():process.terminate()
        for process in processes.values():process.wait(timeout=30)
        for log in logs:log.close()


if __name__=='__main__':main()
