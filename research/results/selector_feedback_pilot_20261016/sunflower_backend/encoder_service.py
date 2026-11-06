"""Bounded canonical A6000 observation service for Sunflower predictor workers."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np

B = Path(__file__).resolve().parent
R = B.parent
Q = B / 'encoding_queue'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.save(f, value)
    tmp.replace(path)


def request(method, frames):
    assert method in ['initial', 'rollout', 'scene_measure']
    Q.mkdir(exist_ok=True)
    key = uuid.uuid4().hex
    source, target = Q / f'{key}.input.npy', Q / f'{key}.output.npy'
    done, error = Q / f'{key}.done.json', Q / f'{key}.error.json'
    req = Q / f'{key}.request.json'
    save(source, frames)
    write(req, {'method': method, 'plan': sha(R / 'PLAN.json'), 'encoder_code': sha(__file__)})
    start = time.monotonic()
    while not done.exists():
        if error.exists():
            raise RuntimeError(error.read_text())
        if time.monotonic() - start > 900:
            raise TimeoutError(f'Canonical encoder request {key}')
        time.sleep(.1)
    metrics = json.loads(done.read_text())
    assert metrics['plan'] == sha(R / 'PLAN.json') and metrics['encoder_code'] == sha(__file__)
    result = np.load(target)
    assert np.isfinite(result).all()
    if method == 'rollout':
        assert result.shape == (32, 196, 128)
    for path in [source, target, done, Q / f'{key}.request.lock']:
        path.unlink(missing_ok=True)
    return result, metrics


def serve():
    # No predictor/scorer/state/update decisions are executed here.
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    os.environ.pop('DISABLE_ADDMM_CUDA_LT', None)
    import torch
    sys.path.insert(0, str(R / 'code'))
    import selector_feedback_data as data
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1 and 'A6000' in torch.cuda.get_device_name(0)
    data.data.checked()
    encoder = data.data.runtime().m.Encoder(data.data.runtime().m.PCA)
    Q.mkdir(exist_ok=True)
    start = time.monotonic(); last = 0.; count = 0
    gpu = str(torch.cuda.get_device_properties(0).uuid)
    plan = sha(R / 'PLAN.json'); code = sha(__file__)
    while time.monotonic() - start < 8 * 3600 and not (B / 'STOP_ENCODERS.json').exists() and not (R / 'SUMMARY.json').exists():
        if time.monotonic() - last > 10:
            write(B / 'encoder_status' / f'{os.getpid()}.json', {'pid': os.getpid(), 'gpu': gpu,
                  'time': time.time(), 'completed': count, 'plan': plan, 'encoder_code': code})
            last = time.monotonic()
        found = False
        for req in sorted(Q.glob('*.request.json')):
            key = req.name.removesuffix('.request.json')
            with req.with_suffix('.lock').open('w') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if not req.exists():
                    continue
                spec = json.loads(req.read_text())
                assert spec['plan'] == plan and spec['encoder_code'] == code
                try:
                    frames = np.load(Q / f'{key}.input.npy')
                    torch.cuda.synchronize(); clock = time.perf_counter()
                    if spec['method'] == 'scene_measure':
                        with torch.no_grad():
                            pixels = encoder.pixels(frames)
                            encoder.model(pixel_values=torch.stack([pixels, pixels], 2)).last_hidden_state
                        result = np.array(0)
                    else:
                        assert spec['method'] in ['initial', 'rollout']
                        result = getattr(encoder, spec['method'])(frames)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - clock
                    save(Q / f'{key}.output.npy', result)
                except Exception:
                    import traceback
                    write(Q / f'{key}.error.json', {'error': traceback.format_exc()})
                    raise
                req.unlink(); count += 1; found = True
            write(Q / f'{key}.done.json', {'plan': plan, 'encoder_code': code, 'gpu': gpu, 'encode_s': elapsed})
            break
        if not found:
            time.sleep(.1)


def launch():
    # Oprime is also a debug GPU host; exclusively use the two observed free GPUs.
    assert not (B / 'STOP_ENCODERS.json').exists()
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True).splitlines()
    gpu_map = {p[0].strip(): (p[1].strip(), p[2].strip()) for p in (line.split(',') for line in lines)}
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    targets = [gpu_map[str(i)] for i in [4, 5]]
    assert all('A6000' in name and gpu not in busy for gpu, name in targets), 'GPU4/5 no longer free'
    children = []
    try:
        for i, (gpu, _) in zip([4, 5], targets):
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu
            log = (B / f'encoder_gpu{i}.out').open('a')
            child = subprocess.Popen([sys.executable, '-u', __file__, 'serve'], env=env, stdout=log, stderr=subprocess.STDOUT)
            log.close(); children.append(child)
        write(B / 'ENCODER_SERVICES.json', {'controller_pid': os.getpid(), 'pids': [p.pid for p in children],
              'physical_gpus': [4, 5], 'time': time.time(), 'max_runtime_hours': 8, 'encoder_code': sha(__file__)})
        while not all(p.poll() is not None for p in children):
            assert all(p.poll() in [None, 0] for p in children), 'Encoder service failed'
            time.sleep(2)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()


if __name__ == '__main__':
    if sys.argv[1] == 'serve': serve()
    elif sys.argv[1] == 'launch': launch()
    else: raise ValueError(sys.argv[1])
