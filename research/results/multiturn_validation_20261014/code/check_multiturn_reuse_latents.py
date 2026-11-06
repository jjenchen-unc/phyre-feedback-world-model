"""Check immutable latent reuse on shared pack samples and extra cached files."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/path/to/phyre-unc')
DATA=Path('/mnt/meg/jjenchen/phyre-unc')
RUN=ROOT/'results/multiturn_validation_20261014'
inventory=json.loads((RUN/'data_reuse.json').read_text());assert inventory['complete']
out=RUN/'reuse_latent_check.json';assert not out.exists()
plan=json.loads((ROOT/'results/small_official_20261010/PLAN.json').read_text())
pca=DATA/'latents/levjepa_overlap_g32_pca128/pca.pt'
pca_sha=hashlib.sha256(pca.read_bytes()).hexdigest()
assert pca_sha==plan['pca']['sha256']
result={'complete':False,'pca_sha256':pca_sha,'shared_samples':[],
        'scope':'Sampled pack equality; all extra cached-file headers checked. No new fold packing performed.'}
for recipe in ['s16r16','clusters']:
    packs=[]
    for panel,fold in [('cross',1),('within',0)]:
        prefix=DATA/'packs'/f'{panel}_f{fold}_train_{recipe}_pca128'
        with np.load(str(prefix)+'_meta.npz') as d:
            by={Path(str(f)).stem:i for i,f in enumerate(d['files'])}
        packs.append((by,np.load(str(prefix)+'_latents.npy',mmap_mode='r')))
    keys=sorted(set(packs[0][0]) & set(packs[1][0]))
    for i in np.linspace(0,len(keys)-1,16,dtype=int):
        key=keys[i];a=packs[0][1][packs[0][0][key]];b=packs[1][1][packs[1][0][key]]
        np.testing.assert_array_equal(a,b)
        standalone=pca.parent/(key+'.npy')
        checked=False
        if standalone.is_file():
            np.testing.assert_array_equal(a,np.load(standalone));checked=True
        result['shared_samples'].append({'recipe':recipe,'key':key,'cross_within_exact':True,
                                         'standalone_checked':checked})
checked=0
for task,action in inventory['union']['missing_keys']:
    p=pca.parent/f'{task.replace(":","_")}_{action:06d}.npy'
    if p.is_file():
        a=np.load(p,mmap_mode='r');assert a.shape==(32,196,128) and a.dtype==np.float16
        checked+=1
assert checked==inventory['union']['standalone_latent_files']
result.update(extra_latent_headers_checked=checked,complete=True)
out.write_text(json.dumps(result,indent=2)+'\n')
print('shared samples',len(result['shared_samples']),'extra headers',checked)
