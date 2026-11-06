"""Restore required CUDA workspace setting; gate on unchanged forward results."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'code'))

def snapshot(tag):
    import numpy as np
    import torch
    import selector_feedback_run as run
    torch.set_num_threads(2)
    plan=run.data.checked();out={}
    encoder=run.LocalEncoder();worker=run.data.CPUWorker()
    try:
        for panel,spec in plan['panels'].items():
            task=spec['train_tasks'][0];pool=run.data.load_pool(panel,task)
            model,probe,head,rule=run.old.load_model(panel,'q0',plan)
            scorer=run.PreparedScorer(head,rule)
            z=torch.from_numpy(pool['z0'][:32].copy()).cuda().float()
            a=torch.from_numpy(pool['action'][:32]).cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(z,a,model.initial_state().expand(32,-1,-1))
            out[panel+'_pred']=pred.float().cpu().numpy()
            out[panel+'_scores']=scorer(pred,[f[None] for f in pool['features'][:32]]).cpu().numpy()
            raw=R/'environment_check'/f'{panel}_{tag}_raw.npz'
            worker.request(mode='rollout',task=task,action=pool['simulation_action'][0].tolist(),out=str(raw))
            with np.load(raw) as data:
                out[panel+'_initial']=encoder.initial(data['frames'][:1])
                out[panel+'_rollout']=encoder.rollout(data['frames'])
            np.testing.assert_array_equal(out[panel+'_initial'][0],pool['z0'][0])
            del model,probe,head,scorer,pred,z,a
            torch.cuda.empty_cache();raw.unlink()
        np.savez(R/'environment_check'/f'{tag}.npz',**out)
    finally:worker.close();encoder.close()

def main():
    import numpy as np
    import selector_feedback_launch as launch
    gpus=launch.gpu_inventory();(R/'environment_check').mkdir(exist_ok=True)
    for tag,config in [('original',None),('configured',':4096:8')]:
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpus[0]
        env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        if config:env['CUBLAS_WORKSPACE_CONFIG']=config
        subprocess.run([launch.ML,str(Path(__file__).resolve()),tag],env=env,check=True)
    with np.load(R/'environment_check/original.npz') as a,np.load(R/'environment_check/configured.npz') as b:
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        checks=list(a.files)
    launch.write(R/'CUDA_ENV_CHECK.json',{'passed':True,'exact_checks':checks,'time':time.time(),'CUBLAS_WORKSPACE_CONFIG':':4096:8','scope':'First32 predictor scores and one initial/full observation perpanel; existing full preflight checks still required'})
    env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    subprocess.run([launch.ML,'-u',str(R/'code/selector_feedback_launch.py'),'pilot'],env=env,check=True)

if __name__=='__main__':
    try:
        if len(sys.argv)>1:snapshot(sys.argv[1])
        else:main()
    except Exception:
        (R/'FAILED.json').write_text(json.dumps({'phase':'cuda_environment_recovery','error':traceback.format_exc(),'time':time.time()}))
        raise
