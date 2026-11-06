"""Bounded dependency chain; no training/evaluation proceeds through a failed gate."""
from adapt_runtime import *
import subprocess


def main():
    checked()
    assert not (R/'JOBS.json').exists()
    jobs = {}
    def submit(name, args, dep=None):
        cmd = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--job-name=probe_'+name,
               '--output='+str(R/(name+'_%A_%a.out'))]
        if dep:
            cmd += ['--dependency='+dep]
        cmd += list(map(str, args))
        j = subprocess.check_output(cmd, cwd=P, text=True).strip().split(';')[0]
        assert j.isdigit()
        jobs[name] = j
        write(R/'JOBS.json', jobs)
        print(name, j, flush=True)
        return j
    cpu, gpu, code = R/'cpu.sbatch', R/'gpu.sbatch', R/'code'
    enc = submit('encoders', [cpu, code/'validate_encoders.py'])
    pre = submit('preflight', [gpu, 'preflight'], 'afterok:'+enc)
    pools = submit('pools', ['--array=0-15', gpu, 'pools'], 'afterok:'+pre)
    cache_smoke = submit('cache_smoke', [gpu, 'cache_smoke'], 'afterok:'+pools)
    train_smoke = submit('train_smoke', ['--array=0-1', gpu, 'train_smoke'], 'afterok:'+cache_smoke)
    smoke = submit('smoke_audit', [cpu, code/'summarize.py', 'smoke'], 'afterok:'+train_smoke)
    collect = submit('collect', ['--array=0-15', gpu, 'collect'], 'afterok:'+smoke)
    data = submit('data_audit', [cpu, code/'collect.py', '--audit'], 'afterok:'+collect)
    train = submit('train', ['--array=0-1', gpu, 'train'], 'afterok:'+data)
    heads = submit('head_audit', [cpu, code/'summarize.py', 'heads'], 'afterok:'+train)
    fixed = submit('fixed', ['--array=0-15', gpu, 'fixed'], 'afterok:'+heads)
    decision = submit('decision', [cpu, code/'summarize.py', 'decision'], 'afterok:'+fixed)
    closed = submit('closed', ['--array=0-31%24', gpu, 'closed'], 'afterok:'+decision)
    final = submit('final', [cpu, code/'summarize.py', 'final'], 'afterok:'+closed)
    submit('cleanup', [cpu, code/'cleanup.py'], 'afterany:'+':'.join(jobs.values()))


if __name__ == '__main__':
    main()
