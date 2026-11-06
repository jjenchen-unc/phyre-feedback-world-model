"""Resume oracle scores unchanged; verify rare initial-rounding cases exactly."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

P=Path('/path/to/phyre-unc')
R=P/'results/milestone_oracle_latent_20261015'
HERE=R/'recovery_20261016'
sys.path.insert(0,str(R/'code'))
spec=importlib.util.spec_from_file_location('oracle_reference',R/'code/milestone_oracle_latent.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
sys.path.insert(0,str(P/'results/deploy_pool_updater_20261016/code'))
import deploy_pool_data as data
encoder=None;worker=None;log=None


def initial_guard(task,ai,action,status,observed,expected,raw):
    global encoder,worker
    diff=np.abs(observed.astype(float)-expected)
    if diff.max()>=.01 or diff.mean()>=1e-6:
        if raw is None:
            if worker is None:worker=data.CPUWorker()
            raw=HERE/'temporary'/f'{os.getpid()}_{task.replace(":","_")}_{ai}.npz'
            response=worker.request(mode='rollout',task=task,action=action.tolist(),out=str(raw))
            assert response['status']==status
        with np.load(raw) as z:
            assert str(z['task'])==task and int(z['status'])==status
            np.testing.assert_array_equal(z['action'],action);frames=z['frames'][:1].copy()
        if encoder is None:
            m=data.runtime().m;encoder=m.Encoder(m.PCA)
        canonical=encoder.initial(frames)[0]
        np.testing.assert_array_equal(canonical,expected)
        log.write(json.dumps({'task':task,'action_idx':ai,'max':float(diff.max()),'mean':float(diff.mean()),
                              'actual_first_frame_reencoded_exact':True})+'\n');log.flush()
    if raw is not None and Path(raw).exists():Path(raw).unlink()


def install():
    tree=ast.parse((R/'code/milestone_oracle_latent.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    class Guard(ast.NodeTransformer):
        changed=0;unlinks=0
        def visit_Assert(self,node):
            if 'diff.max()' in ast.unparse(node.test) and 'diff.mean()' in ast.unparse(node.test):
                self.changed+=1
                return ast.parse('_initial_guard(task,int(indices[i]),actions[i],int(labels[i]),truth[0],initial[i],raw)').body[0]
            return node
        def visit_Expr(self,node):
            if ast.unparse(node)=='raw.unlink()':self.unlinks+=1;return None
            return self.generic_visit(node)
    guard=Guard();module=ast.Module(body=[function],type_ignores=[]);guard.visit(module)
    assert guard.changed==1 and guard.unlinks==1
    original.__dict__['_initial_guard']=initial_guard
    exec(compile(ast.fix_missing_locations(module),str(R/'code/milestone_oracle_latent.py'),'exec'),original.__dict__)


def main():
    global log
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['eval','summary','check','validate'])
    ap.add_argument('--panel');ap.add_argument('--shard',type=int,default=0);a=ap.parse_args()
    original.checked();install()
    if a.mode=='check':print('GUARD_PATCH_TWO_TARGETS_PASS');return
    if a.mode=='validate':
        m=data.runtime().m
        try:
            with (HERE/'validation.jsonl').open('w',buffering=1) as log:
                for panel,task,ai in [('cross','00008:197',3393),('within','00001:001',4232)]:
                    pool,status,_=m.input_case(panel,task);i=int(np.flatnonzero(pool['action_idx']==ai)[0])
                    path=next((data.R/'observations'/panel).glob(f'*/{task.replace(":","_")}_{ai:06d}.npz'))
                    with np.load(path) as z:truth=z['latent'].copy()
                    initial_guard(task,ai,pool['action'][i],int(status[i]),truth[0],pool['z0'][i],None)
            original.write(HERE/'VALIDATION.json',{'passed':True,'cases':2,'wrapper_sha256':original.sha(__file__)})
            print('RECOVERY_CANONICAL_VALIDATION_PASS');return
        finally:
            if worker:worker.close()
    registration={'source_sha256':original.sha(R/'code/milestone_oracle_latent.py'),
                  'wrapper_sha256':original.sha(__file__),
                  'change':'Validation only: exact single-frame reencoding on every old-tolerance exceedance; scored latents and head unchanged'}
    original.write(HERE/'REGISTRATION.json',registration)
    if a.mode=='summary':
        original.summary();result=original.read(R/'SUMMARY.json')
        result['validation_recovery']=registration;original.write(R/'SUMMARY.json',result);return
    logpath=HERE/f'{a.panel}_{a.shard}_initial_checks.jsonl';logpath.parent.mkdir(parents=True,exist_ok=True)
    try:
        with logpath.open('a',buffering=1) as log:
            original.evaluate(a.panel,a.shard,3)
        original.write(HERE/f'{a.panel}_{a.shard}_complete.json',dict(registration,complete=True,time=time.time()))
    finally:
        if worker:worker.close()


if __name__=='__main__':main()
