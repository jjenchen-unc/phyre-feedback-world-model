"""Shared store and paired sampler for the registered deployment pilot."""
from pathlib import Path
import json
import numpy as np
from redesign_lib import Store, ROOT
from redesign_sampling import sample_episode

PILOT_PACK='memory_deploy_pilot_20260929'


class PilotStore(Store):
    def __init__(self, include_features=True):
        super().__init__(include_features=include_features)
        self.old_n=len(self.refs)
        with np.load(ROOT/'packs'/f'{PILOT_PACK}_meta.npz') as p:
            m={k:p[k] for k in p.files if include_features or k not in ['features','mask']}
        self.meta.append(m);self.lat.append(np.load(ROOT/'packs'/f'{PILOT_PACK}_latents.npy',mmap_mode='r'))
        seen={str(f):i for i,f in enumerate(self.files)}
        for j,f in enumerate(m['files']):
            if str(f) in seen:
                i=seen[str(f)];assert self.labels[i]==m['solved'][j] and np.array_equal(self.actions[i],m['action'][j])
            else:self.refs.append((len(self.meta)-1,j));seen[str(f)]=len(self.refs)-1
        self.files=np.array([self.meta[p]['files'][i] for p,i in self.refs])
        self.tasks=np.array([self.meta[p]['tasks'][i] for p,i in self.refs])
        self.labels=np.array([self.meta[p]['solved'][i] for p,i in self.refs],bool)
        self.actions=np.stack([self.meta[p]['action'][i] for p,i in self.refs])


def episode(data,old,store,step,seed,variant):
    rng=np.random.default_rng([20260929,seed,step])
    if rng.random()<.5:
        e=sample_episode(old,store,step,seed,'R0');e['source']='legacy_R0';return e
    t=data['eligible_train_tasks'][int(rng.integers(len(data['eligible_train_tasks'])))];es=data['tasks'][t]['episodes']
    e=es[int(rng.integers(len(es)))];q=rng.choice(e['query'],min(8,len(e['query'])),replace=False).tolist()
    return {'task':t,'context':e[variant],'query':q,'source':'deploy_pilot','attempt':e['attempt'],'actual_k':e['actual_k']}
