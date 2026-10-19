"""Deploy-identical candidate construction and bounded immutable truth caches."""
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

P = Path('/path/to/phyre-project')
R = P / 'results/deploy_pool_updater_20261016'
SOURCE_R = P / 'results/selector_feedback_pilot_20261016'
OLD = P / 'results/milestone_memory_20261015'
PHYRE = Path('/path/to/phyre-python')
LIBC = P / 'runtime/sunflower_glibc'
BUCKET_LIMIT = 4 * 2**30  # 32 buckets per panel = 128 GiB, new run only.


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''): h.update(part)
    return h.hexdigest()
def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n'); temp.replace(path)
def save_npz(path, **data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    with temp.open('wb') as f: np.savez_compressed(f, **data)
    temp.replace(path)
def save_npy(path, data):
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    with temp.open('wb') as f: np.save(f, data)
    temp.replace(path)


def runtime():
    # Freeze the proven loader/evaluator instead of altering historical runs.
    sys.path.insert(0, str(OLD / 'code'))
    import milestone_memory as old
    return old


def prepare():
    assert not (R / 'PLAN.json').exists()
    parent = read(OLD / 'PLAN.json'); plan = copy.deepcopy(parent)
    plan.update(date='2026-10-16', scope='Deploy-pool updater retraining, same Q0 parents and terminal loss',
                parent_plan=str(OLD / 'PLAN.json'), parent_plan_sha256=sha(OLD / 'PLAN.json'),
                methods=['residual', 'zero'], gpu_limit=None)
    plan.pop('gpu_uuids', None)
    plan['dependencies'][str(OLD / 'PLAN.json')] = sha(OLD / 'PLAN.json')
    for path in sorted((OLD / 'code').glob('*.py')):
        plan['dependencies'][str(path)] = sha(path)
    for panel, spec in plan['panels'].items():
        source = read(spec['source_manifest'])
        assert sha(spec['source_manifest']) == spec['source_manifest_sha256']
        tasks = sorted(r['task'] for r in source['rows'] if r['split'] == 'train')
        assert len(tasks) == 1200 and not set(tasks) & set(spec['tasks'])
        spec['train_tasks'] = tasks
        if panel == 'cross': assert not {t[:5] for t in tasks} & {t[:5] for t in spec['tasks']}
    m = runtime().m
    for path in list(m.BASES.values()) + [m.PCA]:
        plan['dependencies'][str(path)] = sha(path)
    plan['training'].update(
        sampling='Same 1200 train scenes; first5000 valid cache actions -> historical base+panel scorer top1024; no outcome balancing',
        queries='8 uniformly sampled untried actions from the SAME 1024 pool after rollout; no label filtering',
        support_caps=[1, 2, 4, 8], slots=10000, episodes_per_slot=2,
        residual='Observed minus current token-conditioned prediction; fresh each turn and full BPTT replay',
        loss='Final updated-state future latent L1 + 0.1 geometry; same as prior milestone',
        empty_episode='No updater gradient; retained in batch weight and scheduler; query loss omitted from diagnostics',
        checkpoint_blocks=False, checkpointing='Whole predictor calls, unchanged full BPTT',
        query_times='8 uniformly sampled future time steps per query',
        distributed='Frozen reader replicas rank disjoint ORIGINAL 32-candidate chunks; current tokens broadcast every turn',
        random_seed=0, initializer_seed=7919, observation_cache_GiB_per_panel=128)
    plan['pool'] = {'initial_actions':5000, 'retained':1024, 'selector':'Historical evaluation shortlist base and panel scorer',
                    'tie_break':'stable original cache action order', 'labels':'Re-simulated after shortlist; never used in selection',
                    'query_support_disjoint':'Exclude every tried action ID and exact action tuple',
                    'initial_encoding':'Historical single-image kernel, float16 PCA128; same as evaluation'}
    plan['runtime_libraries'] = {str(f):sha(f) for f in sorted(LIBC.iterdir()) if f.is_file()}
    write(R / 'PLAN.json', plan)
    print('PREPARED', {p:len(s['train_tasks']) for p,s in plan['panels'].items()}, flush=True)


def checked():
    plan = read(R / 'PLAN.json')
    for path, digest in plan['dependencies'].items(): assert sha(path) == digest, path
    return plan


def encoder_process(connection):
    # A5000 cuBLASLt chooses a different FP16 MLP kernel than historical A6000.
    # Scope this numerical control to the encoder process: applying it to the
    # predictor/probe also changes scores. Never change the parent's environment.
    os.environ['DISABLE_ADDMM_CUDA_LT']='1'
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
    try:
        import torch
        torch.set_num_threads(2)
        m=runtime().m;encoder=m.Encoder(m.PCA)
        connection.send(('ready',None))
        while True:
            method,values=connection.recv()
            if method=='close':break
            assert method in ['initial','rollout']
            connection.send(('ok',getattr(encoder,method)(values)))
    except Exception:
        import traceback
        connection.send(('error',traceback.format_exc()))
        raise
    finally:connection.close()


class EncoderProxy:
    def __init__(self,pca=None):
        self.connection=None;self.process=None
    def start(self):
        ctx=multiprocessing.get_context('spawn')
        self.connection,child=ctx.Pipe()
        self.process=ctx.Process(target=encoder_process,args=(child,),daemon=True)
        self.process.start();child.close()
        status,value=self.connection.recv()
        if status!='ready':raise RuntimeError(value)
    def call(self,method,values):
        if self.process is None:self.start()
        self.connection.send((method,values))
        status,result=self.connection.recv()
        if status!='ok':raise RuntimeError(result)
        return result
    def initial(self,codes):return self.call('initial',codes)
    def rollout(self,frames):
        from deploy_encoder_service import request
        return request(frames)
    def close(self):
        if self.process is None:return
        if self.process.is_alive():self.connection.send(('close',None));self.process.join(timeout=30)
        if self.process.is_alive():self.process.terminate();self.process.join()
        self.connection.close()


class CPUWorker:
    def __init__(self):
        command = [str(LIBC / 'ld-linux-x86-64.so.2'), '--library-path',
                   str(LIBC) + ':' + str(PHYRE.parent.parent / 'lib'), str(PHYRE), '-u',
                   str(R / 'code/deploy_pool_cpu.py')]
        env = os.environ.copy()
        env.update(XDG_CACHE_HOME='/path/to/cache', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1, env=env)
    def request(self, **kwargs):
        self.process.stdin.write(json.dumps(kwargs) + '\n'); self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line: raise RuntimeError('PHYRE worker exited')
            try: result = json.loads(line)
            except json.JSONDecodeError: continue
            assert result['ok'], result
            return result
    def close(self):
        self.process.stdin.close(); self.process.wait(timeout=60)


def folder_for(panel, task, validation=False):
    return R / ('validation_pools' if validation else 'pools') / panel / task.replace(':', '_')


def load_pool(panel, task, validation=False):
    assert not validation
    folder = SOURCE_R / 'pools' / panel / task.replace(':', '_')
    receipt = read(folder / 'complete.json')
    assert receipt['plan'] == sha(SOURCE_R / 'PLAN.json') and receipt['task'] == task
    with np.load(folder / 'pool.npz') as z: data = {k:z[k] for k in z.files}
    data['z0'] = np.load(folder / 'z0.npy', mmap_mode='r')
    assert data['z0'].shape == (1024,196,128) and len(data['action']) == 1024
    assert len(set(map(int, data['action_idx']))) == 1024
    assert len({tuple(a) for a in data['action']}) == 1024
    assert (data['status'] != 0).all()
    return data


def build_pool(panel, task, encoder, base, head, rule, worker, validation=False):
    import torch
    from fast_inference import PreparedScorer
    folder = folder_for(panel, task, validation); folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / 'complete.json').exists(): return load_pool(panel, task, validation)
        if shutil.disk_usage(R).free < 40 * 2**30: raise RuntimeError('Less than 40 GiB free on run filesystem')
        start = time.time(); raw = folder / 'candidates.tmp.npz'
        worker.request(mode='candidates', task=task, out=str(raw))
        with np.load(raw) as z: data = {k:z[k] for k in z.files}
        latents, scores = [], []; scorer = PreparedScorer(head, rule)
        for begin in range(0, len(data['action']), 32):
            end = min(begin+32, len(data['action']))
            latent = encoder.initial(data['frames'][begin:end]); latents.append(latent)
            z0 = torch.from_numpy(latent).cuda().float()
            actions = torch.from_numpy(data['action'][begin:end]).cuda()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): pred = base(z0, actions)
            scores.append(scorer(pred, [f[None] for f in data['features'][begin:end]]).cpu().numpy())
        scores = np.concatenate(scores); latent = np.concatenate(latents)
        assert np.isfinite(scores).all()
        ids = np.argsort(-scores, kind='stable')[:1024]
        labels = folder / 'labels.tmp.npy'
        worker.request(mode='labels', task=task, actions=data['action'][ids].tolist(), out=str(labels))
        status = np.load(labels); assert (status != 0).all()
        save_npy(folder / 'z0.npy', latent[ids])
        save_npz(folder / 'pool.npz', action=data['action'][ids], action_idx=data['action_idx'][ids],
                 features=data['features'][ids], selection_score=scores[ids], status=status)
        write(folder / 'complete.json', {'plan':sha(R / 'PLAN.json'), 'task':task, 'valid':len(scores),
              'candidates':1024, 'positive':int((status>0).sum()), 'seconds':time.time()-start,
              'pool_sha256':sha(folder / 'pool.npz'), 'initial_sha256':sha(folder / 'z0.npy')})
        raw.unlink(); labels.unlink()
        print('POOL', panel, task, 'seconds', time.time()-start, flush=True)
    return load_pool(panel, task, validation)


def collect(shard, nshards, validation=False, panel=None):
    import torch
    plan = checked(); old = runtime(); m = old.m
    torch.set_num_threads(2)
    encoder = EncoderProxy(); worker = CPUWorker()
    try:
        for protocol in ([panel] if panel else ['cross','within']):
            base, head, rule, _ = m.load_base(protocol)
            encoding_metrics=None
            if validation:
                pack='cross_f1_train_s16r16_pca128' if protocol=='cross' else 'within_f0_train_s16r16_pca128'
                with np.load(m.DATA/'packs'/(pack+'_meta.npz')) as z:raw_file=str(z['files'][0])
                raw_path=m.P/'data'/raw_file
                if not raw_path.exists():raw_path=m.DATA/'data'/raw_file
                with np.load(raw_path) as z:frames=z['frames'].copy()
                full=encoder.rollout(frames);initial=encoder.initial(frames[:1])[0]
                cached=np.load(m.DATA/'packs'/(pack+'_latents.npy'),mmap_mode='r')[0]
                encoding_metrics={}
                for label,a,b in [('full_vs_historical_pack',full,cached),('initial_vs_full',initial,full[0])]:
                    diff=np.abs(a.astype(float)-b)
                    encoding_metrics[label]={'max':float(diff.max()),'mean':float(diff.mean())}
                    assert diff.max()<.01 and diff.mean()<1e-6,(label,encoding_metrics[label])
            tasks = plan['panels'][protocol]['train_tasks'][shard::nshards]
            if validation: tasks = plan['panels'][protocol]['tasks'][:1]
            for task in tasks:
                data = build_pool(protocol, task, encoder, base, head, rule, worker, validation)
                if validation:
                    ref, status, _ = m.input_case(protocol, task)
                    for key in ['action_idx','action','z0','features']:
                        np.testing.assert_array_equal(data[key], ref[key], err_msg=key)
                    # NumPy log scores can differ by a float64 ULP across CPUs.
                    # The selected 1024 action IDs AND their order remain exact.
                    score_error=float(np.max(np.abs(data['selection_score']-ref['selection_score'])))
                    np.testing.assert_allclose(data['selection_score'],ref['selection_score'],rtol=0,atol=1e-14)
                    np.testing.assert_array_equal(data['status'], status)
                    write(R / 'preflight' / f'pool_{protocol}.json',
                          {'passed':True, 'plan':sha(R/'PLAN.json'), 'task':task,
                           'all_1024_inputs_order_labels_exact':True,
                           'selection_score_max_abs':score_error,'selection_score_atol':1e-14,
                           'encoder':encoding_metrics,
                           'gpu':torch.cuda.get_device_name(0)})
            del base, head; torch.cuda.empty_cache()
    finally: worker.close(); encoder.close()
    if not validation: write(R/'collection'/f'{shard}.json', {'complete':True,'plan':sha(R/'PLAN.json'),'nshards':nshards})


class TruthCache:
    """Cache observations only, bounded to this run; never cache predictions/residuals."""
    def __init__(self, panel):
        self.panel = panel; self.encoder = None; self.worker = None
        self.root = SOURCE_R/'observations'/panel
    def get(self, task, local, pool):
        ai = int(pool['action_idx'][local]); action = pool['action'][local]
        key = f'{task.replace(":","_")}_{ai:06d}'
        bucket = self.root / str(int(hashlib.sha256(key.encode()).hexdigest(),16)%32)
        bucket.mkdir(parents=True,exist_ok=True); path = bucket/(key+'.npz')
        with (bucket/(key+'.lock')).open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not path.exists():
                if shutil.disk_usage(R).free < 30*2**30: raise RuntimeError('Less than 30 GiB free')
                if self.worker is None: self.worker = CPUWorker()
                if self.encoder is None:
                    self.encoder = EncoderProxy()
                raw = bucket/f'.raw_{os.uname().nodename}_{os.getpid()}.npz'
                response = self.worker.request(mode='rollout', task=task, action=action.tolist(), out=str(raw))
                assert response['status'] == int(pool['status'][local])
                with np.load(raw) as z:
                    latent = self.encoder.rollout(z['frames']); features = z['features'][::4].copy()
                diff = np.abs(latent[0].astype(float)-pool['z0'][local])
                canonical_check=False
                if diff.max()>=.01 or diff.mean()>=1e-6:
                    # Rare FP16 video-vs-single-frame rounding can exceed the
                    # old aggregate tolerance. Require the stronger exact check:
                    # re-encode the ACTUAL rollout's first frame through the
                    # same single-image path that created the candidate latent.
                    with np.load(raw) as z:canonical=self.encoder.initial(z['frames'][:1])[0]
                    np.testing.assert_array_equal(canonical,pool['z0'][local])
                    canonical_check=True
                save_npz(path, latent=latent, features=features, action=action,
                         status=pool['status'][local], task=np.array(task), plan=np.array(sha(SOURCE_R/'PLAN.json')),
                         initial_maxdiff=np.array(diff.max()),initial_meandiff=np.array(diff.mean()),
                         canonical_initial_verified=np.array(canonical_check))
                raw.unlink()
                self.trim(bucket, path)
            with np.load(path) as z:
                assert str(z['plan']) == sha(SOURCE_R/'PLAN.json') and str(z['task']) == task
                assert int(z['status']) == int(pool['status'][local])
                np.testing.assert_array_equal(z['action'],action)
                latent, features = z['latent'].copy(), z['features'].copy()
            os.utime(path,None)
            return {'local':local, 'latent':latent, 'features':features,
                    'status':int(pool['status'][local]), 'action_idx':ai}
    def trim(self, bucket, current):
        # Check at most once a minute per small bucket. Locks protect live readers.
        with (bucket/'trim.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            stamp = bucket/'trim.json'
            if stamp.exists() and time.time()-read(stamp)['time']<60: return
            files = [(f.stat().st_mtime,f.stat().st_size,f) for f in bucket.glob('*.npz') if not f.name.startswith('.')]
            total = sum(x[1] for x in files)
            if total>BUCKET_LIMIT:
                for _,size,path in sorted(files):
                    if path == current: continue
                    with path.with_suffix('.lock').open('w') as live:
                        try: fcntl.flock(live,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError: continue
                        if path.exists(): path.unlink(); total-=size
                    if total<BUCKET_LIMIT*.85: break
            write(stamp,{'time':time.time(),'bytes':total})
    def close(self):
        if self.worker: self.worker.close()
        if self.encoder: self.encoder.close()


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','collect','verify'])
    ap.add_argument('--panel',choices=['cross','within']);ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--nshards',type=int,default=40);a=ap.parse_args()
    if a.mode=='prepare': prepare()
    else: collect(a.shard,a.nshards,a.mode=='verify',a.panel)
