"""Read-only state magnitude diagnostic for all twelve trained updater variants."""
import sys,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
import study as st
spec=importlib.util.spec_from_file_location('norms_train',ROOT/'code/train.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)
import torch
p=st.checked();out=ROOT/'STATE_NORMS.json';assert not out.exists()
rows=[]
for stage,tasks,base in [('fit',p['fit'],ROOT/'data'),('discovery',p['discovery'],ROOT/'data'),('confirmation',p['confirmation'],ROOT/'data'),('cross100',st.read(ROOT/'cross_transfer/EXTENSION_PLAN.json')['all_tasks'],ROOT/'cross_transfer/data')]:
 for task in tasks:
  d=base/task.replace(':','_');m=st.read(d/'complete.json')
  if not m['failed']:continue
  rows.append((stage,task,m['support_action'],np.load(d/'support_observed.npy')-np.load(d/'support_prediction.npy')))
team=st.team_new();reader=st.ar.lr.source().old.reader_hash(team.model);results={}
specs=[(f'{a}_s{seed}',ROOT/'models'/f'seed{seed}'/a,a) for seed,arms in [(7919,st.ARMS),(7920,['prediction_residual','prediction_zero','pair_residual','pair_zero'])] for a in arms]+[(a,ROOT/'calibration_update/models'/a,a) for a in ['bcecal_residual','bcecal_zero']]
try:
 with torch.no_grad():
  for name,d,arm in specs:
   ck=st.read(d/'complete.json');assert ck['checkpoint_sha']==st.sha(d/'updater.pt')
   missing,extra=team.model.load_state_dict(torch.load(d/'updater.pt',map_location='cpu',weights_only=True),strict=False);assert not extra and all(n.startswith('base.') or n=='initial_tokens' for n in missing)
   team.model.eval().requires_grad_(False);team.input_mode='centered' if arm.endswith('residual') else 'zero';initial=team.model.initial_state();rr=[]
   for stage,task,action,residual in rows:
    with torch.autocast('cuda',dtype=torch.bfloat16):new=team.model.update_state(initial,torch.from_numpy(residual).cuda()[None],torch.tensor([action],device='cuda'))
    delta=new-initial
    rr.append(dict(stage=stage,task=task,initial_l2=float(initial.norm()),delta_l2=float(delta.norm()),relative=float(delta.norm()/initial.norm()),new_l2=float(new.norm())))
   assert reader==st.ar.lr.source().old.reader_hash(team.model);results[name]=rr;print('NORMS',name,flush=True)
 summary={name:{stage:{k:float(np.median([r[k] for r in rows if r['stage']==stage])) for k in ['delta_l2','relative','new_l2']} for stage in ['fit','discovery','confirmation','cross100']} for name,rows in results.items()}
 st.write(out,dict(complete=True,parent_plan=st.sha(ROOT/'PLAN.json'),source_sha=st.sha(Path(__file__)),rows=results,median_by_stage=summary,reader_unchanged=True,interpretation='State norm is diagnostic, not proof of a causal failure mechanism. Initial token norm can be small; compare original-loss controls and dose-response rankings before attribution.'))
finally:st.ar.lr.close(team)
