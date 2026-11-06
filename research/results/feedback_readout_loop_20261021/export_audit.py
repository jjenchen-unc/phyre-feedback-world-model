"""Export small heads for independent numerical replay; verify remote receipts."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((R/'PLAN.json').read_text());manifest={}
for seed in p['seeds']:
    for arm in p['arms']:
        d=R/'runs'/f'{seed}_{arm}';r=json.loads((d/'complete.json').read_text())
        assert r['complete'] and r['checkpoint_sha']==sha(d/'head.pt')
        state=torch.load(d/'head.pt',map_location='cpu',weights_only=True)
        np.savez_compressed(d/'weights.npz',**{k:v.numpy() for k,v in state.items()})
for f in R.rglob('*.npz'):manifest[str(f.relative_to(R))]=sha(f)
(R/'REMOTE_HASHES.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('VERIFIED_6_HEADS',len(manifest),flush=True)
