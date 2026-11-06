"""Execution-only recovery: unchanged frozen data code on canonical A6000."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'code'))
import selector_feedback_launch as launch
children=[]
try:
    gpus=launch.gpu_inventory()
    for shard,gpu in enumerate(gpus):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        log=(R/f'recovery_data_{shard}.out').open('a')
        p=subprocess.Popen([launch.ML,'-u',str(R/'code/selector_feedback_data.py'),'collect','--shard',str(shard),'--nshards',str(len(gpus))],env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((p,log))
    while True:
        statuses=[p.poll() for p,_ in children]
        if any(s is not None and s!=0 for s in statuses):raise RuntimeError(f'Collection failed: {statuses}')
        if all(s==0 for s in statuses):break
        time.sleep(5)
    subprocess.run([launch.ML,str(R/'code/selector_feedback_data.py'),'audit'],check=True)
    launch.write(R/'RECOVERY_COMPLETE.json',{'passed':True,'time':time.time(),'frozen_code_unchanged':True,'collection_device':'A6000','workers':len(gpus)})
except Exception:
    launch.write(R/'FAILED.json',{'phase':'a6000_collection','time':time.time(),'error':traceback.format_exc()})
    raise
finally:
    for p,_ in children:
        if p.poll() is None:p.terminate()
    for p,log in children:
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        log.close()
