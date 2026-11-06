"""Real Sunflower-to-Oprime parity, latency and concurrent-client checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys,time
import numpy as np
B=Path(__file__).resolve().parent;R=B.parent
sys.path.insert(0,str(R/'code'))
import selector_feedback_data as pools
from encoder_rpc import Client,sha,write
plan=pools.data.checked();client=Client();cpu=pools.data.CPUWorker();records=[]
try:
    for panel in ['cross','within']:
        refs=sorted((pools.data.SOURCE_R/'observations'/panel).glob('*/*.npz'))[:3]
        assert len(refs)==3
        for k,path in enumerate(refs):
            with np.load(path) as z:
                task=str(z['task']);action=z['action'].copy();expected=z['latent'].copy()
            raw=B/f'rpc_validation_{panel}_{k}.npz'
            cpu.request(mode='rollout',task=task,action=action.tolist(),out=str(raw))
            with np.load(raw) as z:frames=z['frames'].copy()
            actual,metrics=client.request('rollout',frames)
            np.testing.assert_array_equal(actual,expected)
            initial,_=client.request('initial',frames[:1])
            pool=pools.data.load_pool(panel,task)
            ids=np.flatnonzero(np.all(pool['simulation_action']==action,axis=1));assert len(ids)==1
            np.testing.assert_array_equal(initial[0],pool['z0'][ids[0]])
            records.append({'panel':panel,'task':task,'reference_sha256':sha(path),'full_latent_exact':True,'initial_exact':True,**metrics})
            raw.unlink();print('RPC_PARITY',panel,k,metrics,flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        concurrent=list(ex.map(lambda _:Client().request('rollout',frames),range(4)))
    for actual,metric in concurrent:np.testing.assert_array_equal(actual,expected)
    # Detect the old ~30s filequeue overhead independently of legitimate GPU queueing.
    overhead=[r['transport_and_codec_s'] for r in records]+[v[1]['transport_and_codec_s'] for v in concurrent]
    assert max(overhead)<5.,overhead
    write(B/'RPC_VALIDATION.json',{'passed':True,'plan':sha(R/'PLAN.json'),'encoder_rpc':sha(B/'encoder_rpc.py'),
          'records':records,'four_concurrent_clients_exact':True,'concurrent_metrics':[v[1] for v in concurrent],
          'max_transport_and_codec_s':max(overhead),'median_transport_and_codec_s':float(np.median(overhead)),
          'scope':'Six canonical cached trajectories across two panels; fresh simulation and RPC encode, not a hot latent cache. Service shares our oracle GPU during this bounded check.'})
finally:cpu.close()
