"""Register an isolated Q0-first experiment, reusing audited immutable inputs."""
from pathlib import Path
import copy
import hashlib
import json
import time

B = Path(__file__).resolve().parent
R = B.parent
P = B.parents[2]
OLD = P / 'results/selector_feedback_pilot_20261016'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


assert not (R / 'PLAN.json').exists(), 'Do not mutate a registered experiment'
parent = read(OLD / 'PLAN.json')
previous = read(OLD / 'sunflower_backend/TRAINING_SUMMARY.json')
assert previous['training_complete']
for name, digest in previous['provenance']['code'].items():
    assert sha(OLD / 'code' / name) == digest
assert read(OLD / 'DATA_AUDIT.json')['plan'] == sha(OLD / 'PLAN.json')

plan = copy.deepcopy(parent)
first = 'Initial Q0 predictor plus fixed scorer argmax over classifier top1024; same for train/eval/all arms'
plan.update(date='2026-10-18', scope='Q0-first policy; matched retraining versus classifier-first pilot',
            parent_run=str(OLD), parent_plan_sha256=sha(OLD / 'PLAN.json'),
            first_attempt=first, q0='Static Q0 ranking from attempt one; no updates',
            checkpoint_choice='Same user-specified cross lambda0 / within lambda0.03 seed0; fresh updater initialization',
            source_pools=str(OLD / 'pools'), immutable_observation_cache=str(OLD / 'observations'))
plan['training']['first_attempt'] = first
plan['evaluation']['first_attempt'] = first
plan['resources'] = {'training_gpus': 4, 'evaluation_gpu_limit': 16, 'canonical_encoder_gpus': 1}
plan['dependencies'][str(OLD / 'PLAN.json')] = sha(OLD / 'PLAN.json')
for f in sorted((R / 'code').glob('*.py')):
    plan['dependencies'][str(f)] = sha(f)

receipts = []
for panel, spec in plan['panels'].items():
    assert spec['selector'] == 'scene'
    assert len(spec['train_tasks']) == 64 and len(spec['tasks']) == 100
    for task in spec['train_tasks'] + spec['tasks']:
        folder = OLD / 'pools' / panel / task.replace(':', '_')
        receipt = read(folder / 'complete.json')
        assert receipt['plan'] == plan['parent_plan_sha256'] and receipt['task'] == task
        for name, key in [('pool.npz', 'pool_sha256'), ('z0.npy', 'initial_sha256')]:
            assert sha(folder / name) == receipt[key], (panel, task, name)
        receipts.append({'panel': panel, 'task': task, 'receipt_sha256': sha(folder / 'complete.json')})
    print('AUDITED_POOLS', panel, 164, flush=True)
for spec in plan['panels'].values():
    for value in spec['initial'].values():
        assert sha(Path(value['checkpoint'])) == value['sha256']
write(R / 'PLAN.json', plan)
if not (R / 'inputs').exists():
    (R / 'inputs').symlink_to(OLD / 'inputs', target_is_directory=True)
write(R / 'DATA_AUDIT.json', {'passed': True, 'plan': sha(R / 'PLAN.json'),
      'source_plan': plan['parent_plan_sha256'], 'all_328_pool_and_z0_hashes_verified': True,
      'receipts': receipts, 'reuse': 'Immutable classifier pools and canonical observations only; no old predictions, residuals or updater weights'})
policy = read(OLD / 'sunflower_backend/ENCODING_POLICY.json')
policy.update(plan=sha(R / 'PLAN.json'), runtime=sha(B / 'runtime.py'), rpc=sha(B / 'encoder_rpc.py'),
              validation='RPC and reader parity; four fresh Q0-first preflights including exact train/eval first scores and full BPTT')
write(B / 'ENCODING_POLICY.json', policy)
write(R / 'STATUS.json', {'stage': 'registered', 'time': time.time(), 'training_complete': False, 'evaluation_complete': False})
print('REGISTERED', sha(R / 'PLAN.json'), flush=True)
