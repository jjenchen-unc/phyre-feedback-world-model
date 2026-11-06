"""Separate first-scene setup from repeated simulation/save of the same scene."""
import json
import time
import subprocess
from pathlib import Path
import numpy as np

r = Path('/path/to/phyre-unc')
out = r/'results/multiturn_validation_20261014/repeated_simulation.json'
assert not out.exists()
worker = subprocess.Popen(['/path/to/research-assets/miniconda3/envs/phyre/bin/python','-u',
    str(r/'results/small_official_20261010/code/data_cpu.py'),'serve'],
    stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
result = {'complete': False, 'rows': []}
try:
    for panel, fold in [('cross',0),('cross',2),('within',1),('within',2)]:
        path=r/'data/manifests'/f'ball_{panel}_template_f{fold}_train.jsonl'
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        for row in (rows[0],rows[len(rows)//2]):
            with np.load(r/'data'/row['file']) as z:action=z['action'].tolist()
            for repeat in range(4):
                raw=out.parent/'repeat_sim_temp.npz';start=time.perf_counter()
                worker.stdin.write(json.dumps({'task':row['task'],'action':action,'out':str(raw)})+'\n')
                worker.stdin.flush()
                while True:
                    line=worker.stdout.readline()
                    if not line:raise RuntimeError('worker ended')
                    try:response=json.loads(line)
                    except json.JSONDecodeError:continue
                    assert response['ok'];break
                result['rows'].append({'panel':panel,'fold':fold,'task':row['task'],
                    'repeat':repeat,'seconds':time.perf_counter()-start})
                raw.unlink();out.write_text(json.dumps(result,indent=2)+'\n')
    result['complete']=True;out.write_text(json.dumps(result,indent=2)+'\n')
finally:
    worker.stdin.close();worker.wait(timeout=60)
