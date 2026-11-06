"""Post-training dose response on fixed first-feedback histories; no retraining."""
import sys,argparse,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import study as st
spec=importlib.util.spec_from_file_location('amplitude_train',ROOT/'code/train.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)
RR=ROOT/'update_amplitude';read,write,sha=st.read,st.write,st.sha
ARMS=['bce_residual_s7919','bce_zero_s7919','pair_residual_s7919','pair_zero_s7919','pair_residual_s7920','pair_zero_s7920'];ALPHAS=[.1,.25,.5]


def register():
 p=st.checked();cross=read(ROOT/'cross_transfer/PLAN.json')
 plan=dict(parent=sha(ROOT/'PLAN.json'),cross_parent=sha(ROOT/'cross_transfer/PLAN.json'),source_sha=sha(Path(__file__)),tasks=cross['tasks'],arms=ARMS,alphas=ALPHAS,
   trigger='On the completed20-scene cross transfer, success-supervised updates lose early-rank coverage and macroAUC versusQ0/prediction. Training-template gains unstable acrossseeds. Test whether attenuating state delta reverses harm before recommending another loss.',
   intervention='state = initial_tokens + alpha*(trained_update(initial_tokens,currentQ0residual,first_action)-initial_tokens). Same observedfirstfailure, originalscorer, fixedcandidatepool. Include trainedzero controls; alpha0Q0/alpha1 existingrank scores; all3intermediatealphas evaluated without winnerselection.',
   stage='Diagnostic intervention on alreadyexamined20crossscenes, not independentconfirmation of a selectedalpha. No gradient fitting, newlabels, simulationorencoding.',
   interpretation='If attenuation restoresQ0 only, that is harm reduction, not evidence of useful residual. Direction success requires improvement beyondQ0 AND appropriatezero controls, not justbeyondalpha1. A changed inputstate may beOOD; mechanism evidence only.')
 assert not(RR/'PLAN.json').exists();write(RR/'PLAN.json',plan)


def check():
 p=read(RR/'PLAN.json');assert p['parent']==sha(ROOT/'PLAN.json') and p['cross_parent']==sha(ROOT/'cross_transfer/PLAN.json') and p['source_sha']==sha(Path(__file__));st.checked();return p


def evaluate(shard,nshards):
 import torch
 from fast_inference import PreparedScorer
 p=check();team=st.team_new();scorer=PreparedScorer(team.head,team.rule);reader=st.ar.lr.source().old.reader_hash(team.model)
 try:
  for task in p['tasks'][shard::nshards]:
   out=RR/'evaluation'/task.replace(':','_');d=ROOT/'cross_transfer/data'/task.replace(':','_');r=read(d/'complete.json')
   if(out/'result.json').exists():assert read(out/'result.json')['plan']==sha(RR/'PLAN.json');continue
   if not r['failed']:
    write(out/'result.json',dict(complete=True,plan=sha(RR/'PLAN.json'),task=task,first_failed=False));continue
   team.load(task);row=dict(support_action=r['support_action'],support_prediction=np.load(d/'support_prediction.npy'),support_observed=np.load(d/'support_observed.npy'))
   with np.load(ROOT/'cross_transfer/evaluation'/task.replace(':','_')/'scores.npz') as z:ref={k:z[k].copy() for k in z.files}
   np.testing.assert_array_equal(ref['indices'],team.pool['action_idx']);np.testing.assert_array_equal(ref['labels'],team.pool['status']>0)
   arrays={};norms={};anchor_errors={}
   with torch.no_grad():
    for name in p['arms']:
     arm,seed=name.rsplit('_s',1);folder=ROOT/'models'/f'seed{seed}'/arm;ck=read(folder/'complete.json');assert ck['checkpoint_sha']==sha(folder/'updater.pt')
     missing,extra=team.model.load_state_dict(torch.load(folder/'updater.pt',map_location='cpu',weights_only=True),strict=False);assert not extra and all(n.startswith('base.') or n=='initial_tokens' for n in missing)
     team.model.eval().requires_grad_(False);team.model.checkpoint_blocks=False;team.input_mode='centered' if arm.endswith('residual') else 'zero'
     initial=team.model.initial_state();updated=tr.state_for(team,row);delta=updated-initial
     norms[name]=dict(initial_l2=float(initial.norm()),delta_l2=float(delta.norm()),relative=float(delta.norm()/initial.norm()))
     # Check alpha1 andalpha0 exact originalscores on chosenaction's originalchunk.
     start=r['first']//32*32;z,a=team.chunk(start);fs=[f[None] for f in team.pool['features'][start:start+32]]
     for tag,state in [('q0',initial),(name,updated)]:
      with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a,state.expand(32,-1,-1))
      value=scorer(pred,fs).cpu().numpy();np.testing.assert_array_equal(value,ref[tag][start:start+32]);anchor_errors[tag]=float(abs(value-ref[tag][start:start+32]).max())
     for alpha in p['alphas']:
      state=initial+alpha*delta;chunks=[]
      for start in range(0,1024,32):
       z,a=team.chunk(start)
       with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a,state.expand(32,-1,-1))
       chunks.append(scorer(pred,[f[None] for f in team.pool['features'][start:start+32]]).cpu().numpy())
      arrays[f'{name}_a{alpha}']=np.concatenate(chunks)
   assert reader==st.ar.lr.source().old.reader_hash(team.model)
   st.ar.lr.source().data.save_npz(out/'scores.npz',**arrays,labels=ref['labels'],indices=ref['indices'],first=ref['first'])
   write(out/'result.json',dict(complete=True,plan=sha(RR/'PLAN.json'),task=task,first_failed=True,norms=norms,anchor_errors=anchor_errors,reader_unchanged=True));print('AMPLITUDE',task,norms,flush=True)
 finally:st.ar.lr.close(team)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','evaluate']);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=8);a=p.parse_args();register() if a.mode=='register' else evaluate(a.shard,a.nshards)
