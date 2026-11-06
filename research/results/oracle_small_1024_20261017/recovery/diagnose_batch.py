from pathlib import Path
import sys, json
import numpy as np
import torch
P=Path('/path/to/phyre-unc');R=P/'results/oracle_small_1024_20261017'
sys.path.insert(0,str(P/'results/selector_feedback_pilot_20261016/code'))
import selector_feedback_run as run
from fast_inference import PreparedScorer
torch.set_num_threads(2)
plan=run.data.checked();panel='cross';task='00008:144'
pool=run.data.load_pool(panel,task)
model,probe,head,rule=run.old.load_model(panel,'base',plan)
scorer=PreparedScorer(head,rule)
with np.load(P/'results/candidate_coverage_100k_20261016/scores'/panel/'base'/task.replace(':','_')/'scores.npz') as f:
 valid=f['rendered_valid'];reference=f['scores'][pool['action_idx']]
scores=[]
with torch.no_grad():
 for b in range(0,1024,32):
  z=torch.from_numpy(pool['z0'][b:b+32].copy()).cuda().float();a=torch.from_numpy(pool['action'][b:b+32]).cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z,a)
  scores.extend(scorer(pred,[f[None] for f in pool['features'][b:b+32]]).cpu().numpy())
scores=np.array(scores);mismatched=np.flatnonzero(scores!=reference);details=[]
with torch.no_grad():
 for j in mismatched:
  idx=int(pool['action_idx'][j]);start=idx//512*512;ids=np.flatnonzero(valid[start:start+512])+start
  position=int(np.flatnonzero(ids==idx)[0]);size=min(32,len(ids)-position//32*32)
  # Same example repeated: tests batch-shape arithmetic without changing its inputs.
  values={}
  for n in sorted({1,size,32}):
   z=torch.from_numpy(pool['z0'][j:j+1].copy()).cuda().float().repeat(n,1,1)
   a=torch.from_numpy(pool['action'][j:j+1]).cuda().repeat(n,1)
   with torch.autocast('cuda',dtype=torch.bfloat16):p=model(z,a)
   values[str(n)]=float(scorer(p,[pool['features'][j][None]]*n)[0].cpu())
  details.append({'local':int(j),'action_idx':idx,'original_batch_size':size,'archived':float(reference[j]),'current32':float(scores[j]),'batch_scores':values,'original_size_reproduces_exactly':bool(values[str(size)]==reference[j])})
  print(details[-1],flush=True)
result={'task':task,'mismatches':len(mismatched),'max_abs':float(np.abs(scores-reference).max()),'details':details,
        'full_rank_same':bool(np.array_equal(np.argsort(-scores,kind='stable'),np.argsort(-reference,kind='stable')))}
run.data.write(R/'recovery/BATCH_DIAGNOSIS.json',result)
print(json.dumps(result),flush=True)
