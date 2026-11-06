"""Replace only our pending evaluation jobs with per-panel dependencies."""
import json
from pathlib import Path
import subprocess

P = Path('/path/to/phyre-unc')
R = P / 'results/memory_overnight_investigation_20261019'
receipt = R / 'PANEL_SCHEDULING.json'
assert not receipt.exists(), 'Inspect existing scheduling receipt; do not submit duplicates'
old = ['36070', '36071', '36075', '36076']
state = subprocess.check_output(['squeue', '-h', '-j', ','.join(old), '-o', '%i %T'], text=True)
assert len(state.splitlines()) == 4 and all(line.endswith(' PENDING') for line in state.splitlines()), state
out = {'old_jobs': old, 'old_states': state, 'new_jobs': {}, 'complete': False}


def save():
    receipt.write_text(json.dumps(out, indent=2) + '\n')


def submit(name, args):
    job = subprocess.check_output(['sbatch', '--parsable'] + args, cwd=P, text=True).strip().split(';')[0]
    assert job.isdigit()
    out['new_jobs'][name] = job
    save()
    print(name, job, flush=True)
    return job


save()
arrays = {}
for panel, tasks, array in [('cross', '36066_0:36066_1', '0-28:4,1-29:4%32'),
                            ('within', '36066_2:36066_3', '2-30:4,3-31:4%32')]:
    arrays[panel] = submit(panel + '_raw_eval', ['--array=' + array, '--dependency=afterok:' + tasks,
        'results/query_balance_training_20261019/eval.sbatch'])
    arrays[panel + '_centered'] = submit(panel + '_centered_eval', ['--array=' + array,
        '--dependency=afterok:' + tasks + ':36074', 'results/query_balance_centered_eval_20261019/run.sbatch'])
common = ['--partition=defq', '--cpus-per-task=1', '--time=00:10:00']
base = submit('raw_summary', common + ['--job-name=phyre_query_sum',
    '--dependency=afterok:' + arrays['cross'] + ':' + arrays['within'],
    '--output=results/query_balance_training_20261019/summary_%j.out',
    '--wrap=.venv-ml/bin/python results/query_balance_training_20261019/code/evaluate.py summary --slot 128'])
submit('centered_summary', common + ['--job-name=phyre_center_sum',
    '--dependency=afterok:' + arrays['cross_centered'] + ':' + arrays['within_centered'] + ':' + base,
    '--output=results/query_balance_centered_eval_20261019/summary_%j.out',
    '--wrap=.venv-ml/bin/python results/query_balance_centered_eval_20261019/code/evaluate.py summary'])
# Recheck immediately before cancelling: never cancel an evaluation that started.
state = subprocess.check_output(['squeue', '-h', '-j', ','.join(old), '-o', '%i %T'], text=True)
assert len(state.splitlines()) == 4 and all(line.endswith(' PENDING') for line in state.splitlines()), state
subprocess.run(['scancel'] + old, check=True)
out['complete'] = True
out['reason'] = 'Identical evaluation, tasks, checkpoints and budgets; within pair can evaluate before cross training finishes.'
save()
