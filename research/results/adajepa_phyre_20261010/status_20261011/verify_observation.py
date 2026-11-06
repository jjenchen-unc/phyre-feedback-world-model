"""Diagnose, and cache only after exact source/re-encoding checks pass."""
import json,sys,runpy,fcntl
from pathlib import Path
import numpy as np,torch
P=Path('/path/to/phyre-unc');M=P/'results/small_official_20261010';R=P/'results/adajepa_phyre_20261010/status_20261011'
sys.path.insert(0,str(M/'recovery_20261010'));import recovery_runner as m
torch.set_num_threads(4);enc=m.Encoder(m.PCA)
raw=M/'assets_fixed/cross/observations/.raw_3960192_00008_197_3393.npz'
with np.load(raw) as z:frames=z['frames'];features=z['features'];action=z['action'];status=int(z['status']);task=str(z['task'])
d,labels,path=m.input_case('cross',task);j=int(np.flatnonzero((d['action']==action).all(1))[0]);assert int(d['action_idx'][j])==3393 and labels[j]==status==-1
with np.load(M/'assets/cross/candidates'/(task.replace(':','_')+'.npz')) as c:
 i=int(np.flatnonzero(c['action_idx']==3393)[0]);np.testing.assert_array_equal(c['frames'][i],frames[0]);np.testing.assert_array_equal(c['features'][i],features[0])
initial=enc.initial(frames[:1])[0];np.testing.assert_array_equal(initial,d['z0'][j]);full=enc.rollout(frames)
sys.argv=['legacy','--encoder','levjepa','--manifests','unused','--pca',str(m.PCA)]
legacy=runpy.run_path(str(M/'code/legacy_encode_latents.py'),run_name='legacy')
with torch.no_grad():
 h=legacy['encode_rollout'](frames)[list(range(0,128,4))].reshape(32,196,-1)
 reference=enc.project(h).cpu().numpy();np.testing.assert_array_equal(full,reference);true_hidden=h[0].clone()
 # Same B8,T16 encoder shape; perturb every future frame, retain first image.
 x=enc.pixels(frames[:1]);v=x[None].expand(8,16,3,224,224).contiguous()
 h=enc.model(pixel_values=v.permute(0,2,1,3,4)).last_hidden_state[:,1:].reshape(8,16,196,-1)
 assert torch.equal(h[0,0],true_hidden),'Causal first-frame encoder hidden changed'
 repeated=enc.project(h[0,0][None].expand(32,-1,-1).contiguous()).cpu().numpy()[0]
np.testing.assert_array_equal(repeated,full[0])
diff=np.abs(full[0].astype(float)-initial);metrics={'max':float(diff.max()),'mean':float(diff.mean()),'p99':float(np.quantile(diff,.99)),'fraction_nonzero':float((diff>0).mean())}
# This one cache entry is accepted from exact legacy/source equivalence, not from a looser global guard.
assert diff.max()<.01
report={'passed':True,'task':task,'action_idx':3393,'raster_features_exact':True,'initial_vs_shortlist_exact':True,'full_vs_legacy_exact':True,'future_perturbation_hidden_exact':True,'matched_projection_shape_t0_exact':True,'difference':metrics,'raw_sha256':m.sha(raw),'verifier_sha256':m.sha(__file__),'policy':'one verified observation; unchanged runtime guards and unchanged encoded target values'}
print(json.dumps(report,indent=2),flush=True)
out=M/'assets_fixed/cross/observations/00008_197_003393.npz'
with open(str(out)+'.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert not out.exists()
 m.f.save_npz(out,{'latents':full,'features':features[::4],'action':action,'status':np.array(status),'task':np.array(task),'pca_sha256':np.array(m.sha(m.PCA)),'initial_encoding_maxdiff':np.array(metrics['max']),'verification_sha256':np.array(m.sha(__file__))})
report['cache_sha256']=m.sha(out);m.write(R/'OBSERVATION_VERIFIED.json',report)
