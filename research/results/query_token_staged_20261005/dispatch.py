"""Map fixed scheduler arrays to registered seeds/branches/splits."""
import os
import sys
from pathlib import Path
ROOT=Path('/path/to/phyre-unc/results/query_token_staged_20261005')
sys.path.insert(0,str(ROOT/'code'))
import query_token_staged as s
s.torch.set_num_threads(4)
command=sys.argv[1];index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
if command in ['q0_train','residual_train','zero_train']:
    s.train(ROOT,command.removesuffix('_train'),index)
elif command=='q0_eval':s.evaluate(ROOT,'q0',index%3,index>=3)
elif command=='updater_eval':s.evaluate(ROOT,['residual','zero'][(index//3)%2],index%3,index>=6)
elif command=='training_audit':
    m=s.checked(ROOT);models={}
    for seed in m['seeds']:
        rr=[s.f.read(ROOT/'models'/f'{stage}_s{seed}'/'complete.json') for stage in ['q0','residual','zero']]
        for r in rr:
            assert r['provenance']==s.f.provenance(ROOT)
            assert s.f.digest(r['checkpoint'])==r['checkpoint_sha256']
            assert r['steps']==(52272 if r['stage']=='q0' else 10000)
            assert r['checks']['frozen_unchanged'] and r['checks']['initial_identity']
        assert rr[1]['parent_checkpoint_sha256']==rr[2]['parent_checkpoint_sha256']==rr[0]['checkpoint_sha256']
        assert rr[1]['frozen_sha256']==rr[2]['frozen_sha256']
        assert rr[1]['draw_sha256']==rr[2]['draw_sha256']
        assert rr[1]['initial_model_sha256']==rr[2]['initial_model_sha256']
        models[str(seed)]={r['stage']:r['checkpoint_sha256'] for r in rr}
    s.f.write_json(ROOT/'TRAINING_AUDIT.json',{'passed':True,'provenance':s.f.provenance(ROOT),'checkpoints':models})
elif command in ['cross_summary','within_summary']:
    within=command=='within_summary'
    # Independently reconstruct the static-policy results from saved scores.
    base=ROOT/('within_evaluation' if within else 'evaluation')
    for seed in range(3):
        for path in (base/f'q0_s{seed}'/'closed').glob('*/result.json'):
            r=s.f.read(path)
            with s.np.load(path.parent/'scores.npz') as z:
                for key in ['q0','original']:
                    hits=s.np.flatnonzero(z['labels'][s.np.argsort(-z[key],kind='stable')])
                    rank=int(hits[0])+1 if len(hits) else None
                    assert rank==r[key+'_first_rank']
    s.summarize(ROOT,within)
else:raise ValueError(command)
