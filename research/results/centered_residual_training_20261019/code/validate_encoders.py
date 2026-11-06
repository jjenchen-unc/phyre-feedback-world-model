from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,sys
import numpy as np
P=Path('/path/to/phyre-unc');S=P/'results/q0_first_feedback_20261018';R=P/'results/centered_residual_training_20261019'
sys.path[:0]=[str(S/'code'),str(S/'sunflower_backend'),str(P/'pipeline')]
import selector_feedback_data as pools
from encoder_rpc import Client
clients=[Client(R/f'encoder{i}/rpc/ENDPOINT.json') for i in range(2)]
cpu=pools.data.CPUWorker();records=[]
try:
    for panel in ['cross','within']:
        task=json.loads((P/'results/residual_mean_step_diagnostic_20261019/PLAN.json').read_text())['tasks'][panel][0]
        row=json.loads((S/'evaluation'/panel/'residual'/task.replace(':','_')/'result.json').read_text());j=row['events'][0]['local'];pool=pools.data.load_pool(panel,task)
        key=f'{task.replace(":","_")}_{int(pool["action_idx"][j]):06d}';path=pools.data.SOURCE_R/'observations'/panel/str(int(hashlib.sha256(key.encode()).hexdigest(),16)%32)/(key+'.npz')
        with np.load(path) as z:expected=z['latent'].copy();action=z['action'].copy()
        raw=R/f'encoder_validation_{panel}.npz';cpu.request(mode='rollout',task=task,action=action.tolist(),out=str(raw))
        with np.load(raw) as z:frames=z['frames'].copy()
        with ThreadPoolExecutor(max_workers=2) as ex:values=list(ex.map(lambda c:c.request('rollout',frames),clients))
        for i,(actual,metrics) in enumerate(values):
            np.testing.assert_array_equal(actual,expected)
            initial,_=clients[i].request('initial',frames[:1]);np.testing.assert_array_equal(initial[0],pool['z0'][j])
            records.append({'panel':panel,'task':task,'encoder':i,'latent_exact':True,'initial_exact':True,'encode_s':metrics['encode_s']})
        raw.unlink();print('ENCODERS_EXACT',panel,flush=True)
    (R/'ENCODER_VALIDATION.json').write_text(json.dumps({'passed':True,'source_rpc_sha256':hashlib.sha256((S/'sunflower_backend/encoder_rpc.py').read_bytes()).hexdigest(),'records':records},indent=2)+'\n')
finally:cpu.close()
