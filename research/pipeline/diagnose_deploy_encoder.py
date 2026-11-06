import sys
from pathlib import Path
sys.path.insert(0,'/path/to/phyre-unc/results/deploy_pool_updater_20261016/code')
import deploy_pool_data as d
import numpy as np
import torch

m=d.runtime().m;p=d.checked();task=p['panels']['cross']['tasks'][0];enc=m.Encoder(m.PCA)
with np.load(d.R/'raw_diagnostic.npz') as f: a={k:f[k] for k in f.files}
ref,status,_=m.input_case('cross',task);idx=int(np.flatnonzero(a['action_idx']==ref['action_idx'][0])[0])
records={}
def hook(name):
    def save(module,inputs,out):
        if isinstance(out,torch.Tensor):records[name]=out.detach().cpu()
    return save
for name,module in enc.model.named_modules():
    if name.startswith('encoder.patch_embed') or name.startswith('encoder.blocks.0.') or name in ['encoder.blocks.0','encoder.blocks.1','encoder.blocks.23']:
        module.register_forward_hook(hook(name))
records['pixels']=enc.pixels(a['frames'][idx:idx+1]).cpu()
records['output']=torch.from_numpy(enc.initial(a['frames'][idx:idx+1]))
torch.save(records,d.R/('encoder_hooks_'+sys.argv[1]+'.pt'))
print('SAVED',sys.argv[1],flush=True)
