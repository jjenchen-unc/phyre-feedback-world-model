"""Validate every evaluation result, summarize, then release our encoder."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

B = Path(__file__).resolve().parent
R = B.parent
P = B.parents[2]
os.chdir(P)
training = (json.loads((B / 'TRAINING_SUMMARY.json').read_text())
            if (B / 'TRAINING_SUMMARY.json').exists() else
            {'training_complete': False, 'evaluation_complete': False})
status = dict(training, stage='evaluation_incomplete', evaluation_complete=False, time=time.time())
try:
    assert training['training_complete'], 'Training or preflight did not complete successfully'
    for panel in ['cross', 'within']:
        for kind in ['q0', 'residual', 'zero', 'tta']:
            for shard in range(4):
                receipt = json.loads((R / 'evaluation' / panel / kind / f'shard_{shard}.json').read_text())
                assert receipt['complete'] and receipt['provenance'] == training['provenance']
    subprocess.run([sys.executable, str(B / 'runtime.py'), 'summary'], check=True)
    path = R / 'SUMMARY.json'
    summary = json.loads(path.read_text())
    summary['timing_limits'] = [
        'Initial 100k ranking costs come from the earlier coverage run; classifier initial-image encoding is separate.',
        'Feedback timings include immutable observation-cache hits and queue/network overhead; not cold deployment latency.',
        'Evaluation uses parallel A5000 predictor workers sharing one canonical A6000 observation encoder on Oprime.',
        'Allocated GPU hours are not CUDA-active kernel time; peak memory reports only the predictor GPU.',
    ]
    path.write_text(json.dumps(summary, indent=2) + '\n')
    status.update(stage='complete', evaluation_complete=True, summary=str(path))
except Exception:
    status['error'] = traceback.format_exc()
finally:
    status['time'] = time.time()
    tmp = R / f'STATUS.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(status, indent=2) + '\n')
    tmp.replace(R / 'STATUS.json')
    (B / 'EVALUATION_COMPLETION.json').write_text(json.dumps(status, indent=2) + '\n')
    (B / 'rpc/STOP').write_text('Evaluation array ended; see EVALUATION_COMPLETION.json.\n')
print(json.dumps({k: v for k, v in status.items() if k not in ['arms', 'provenance']}, indent=2))
if not status['evaluation_complete']:
    raise SystemExit(1)
