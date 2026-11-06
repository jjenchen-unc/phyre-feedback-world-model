"""Bounded existing-rollout encoding and fresh simulation cost samples."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from validate_multiturn_runtime import m, ROOT, Telemetry, timed


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    result={'complete':False,'scope':'bounded cost sample, not full newfold preparation','samples':[],'inventory':[]}
    samples=[]
    for panel,fold in [('cross',0),('cross',2),('within',1),('within',2)]:
        path=ROOT/'data/manifests'/f'ball_{panel}_template_f{fold}_train.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        result['inventory'].append({'panel':panel,'fold':fold,'raw_rollouts':len(rows),'tasks':len({r['task'] for r in rows})})
        # Evenly spaced scene coverage; eight per new panel, no outcome filtering.
        samples.extend((panel,fold,rows[int(i)]) for i in np.linspace(0,len(rows)-1,8,dtype=int))
    with Telemetry() as telemetry:
        def save():result['telemetry']=telemetry.samples.copy();m.write(args.output,result)
        start=time.perf_counter();encoder=m.Encoder(m.PCA);torch.cuda.synchronize()
        result['encoder_load_s']=time.perf_counter()-start;save()
        worker=subprocess.Popen([m.PHYRE,'-u',str(m.R/'code/data_cpu.py'),'serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        try:
            for i,(panel,fold,row) in enumerate(samples):
                start=time.perf_counter()
                with np.load(ROOT/'data'/row['file']) as d:
                    frames=d['frames'];action=d['action']
                read_s=time.perf_counter()-start
                latent,encode_s=timed(lambda:encoder.rollout(frames))
                initial,initial_s=timed(lambda:encoder.initial(frames[:1]))
                record={'panel':panel,'fold':fold,'task':row['task'],'warmup':i<2,'load_raw_s':read_s,
                        'trajectory_encode_s':encode_s,'initial_encode_s':initial_s,'latent_bytes':latent.nbytes,'latent_shape':list(latent.shape)}
                # Fresh simulation uses isolated benchmark paths, not historical caches.
                raw=args.output.parent/f'benchmark_sim_{i}.npz';start=time.perf_counter()
                worker.stdin.write(json.dumps({'task':row['task'],'action':action.tolist(),'out':str(raw.resolve())})+'\n');worker.stdin.flush()
                while True:
                    line=worker.stdout.readline()
                    if not line:raise RuntimeError('Simulation worker exited')
                    try:response=json.loads(line)
                    except json.JSONDecodeError:continue
                    assert response['ok'];break
                record['simulation_save_s']=time.perf_counter()-start
                record['simulated_status']=response['status']
                assert int(response['status'])==int(row['status'])
                raw.unlink()
                result['samples'].append(record);save();print('DATA',i,record,flush=True)
        finally:
            worker.stdin.close();worker.wait(timeout=60)
        result['complete']=True;save()


if __name__=='__main__':main()
