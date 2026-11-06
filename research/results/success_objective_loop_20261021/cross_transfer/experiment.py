"""Fixed 20-scene cross-template transfer panel, identical first-feedback histories."""
import argparse,sys,importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import study as st

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
tr=module('transfer_local_train',ROOT/'code/train.py');ro=module('transfer_readout',ROOT/'readout/experiment_v2.py')
RR=ROOT/'cross_transfer';read,write,sha=st.read,st.write,st.sha


def register():
    prior=st.ar.checked();templates=sorted({t[:5] for t in prior['test']});rng=np.random.default_rng(20261021)
    tasks=[str(t) for template in templates for t in rng.permutation(sorted(t for t in prior['test'] if t.startswith(template)))[:4]]
    assert len(tasks)==20 and not set(templates)&set(prior['templates'])
    plan=dict(parent_plan=sha(ROOT/'PLAN.json'),source_sha=sha(Path(__file__)),readout_source_sha=sha(ROOT/'readout/experiment_v2.py'),readout_models_sha=sha(ROOT/'readout/MODELS.json'),tasks=tasks,templates=templates,arms_seed7919=st.ARMS,arms_seed7920=['prediction_residual','prediction_zero','pair_residual','pair_zero'],calibration_arms=['bcecal_residual','bcecal_zero'],
      selection='Fixed RNG20261021,4uniformscenes per5 officialcrossfold1testtemplates fromexisting100scene panel, no outcome filtering. Previously used testpanel, not claimed untouched.',
      scope='Freeze fittedmethods before newpaneloutcomes. Same classifier1024pool, originalQ0first action, same observedfirstfailure residual and currentQ0 prediction. Single-feedback next1/3/5/8 ranking; no further memory updates. Also fullstaticranking ofQ0,classifier,successreadouts for reference.',
      controls='Q0; initialsceneclassifier within samepool; prediction/BCE/pair real/zero seed7919; prediction/pair real/zero seed7920; learnedaffineBCE real/zero; query8/fullpool successreadouts on frozenQ0 geometry.',
      interpretation='Primary transferpanel for whether early-rank improvements generalize beyond16trainingtemplates. Only20scenes/5templates, two updaterseeds sharebase, so exploratorydirection not definitivebenchmark.',
      data='Reuse exact cached observations; no simulation/encoding fallback or cached evolving-token predictions. Firstprediction fixedbecauseQ0frozen.')
    assert not(RR/'PLAN.json').exists();write(RR/'PLAN.json',plan)


def checked():
    p=read(RR/'PLAN.json');assert p['source_sha']==sha(Path(__file__)) and p['parent_plan']==sha(ROOT/'PLAN.json') and p['readout_source_sha']==sha(ROOT/'readout/experiment_v2.py') and p['readout_models_sha']==sha(ROOT/'readout/MODELS.json');st.checked();return p


def pack(shard,nshards):
    import torch
    from fast_inference import PreparedScorer
    from scorer_torch import distances
    plan=checked();team=st.team_new();team.model.requires_grad_(False);scorer=PreparedScorer(team.head,team.rule)
    try:
        for task in plan['tasks'][shard::nshards]:
            out=RR/'data'/task.replace(':','_')
            if(out/'complete.json').exists():assert read(out/'complete.json')['plan']==sha(RR/'PLAN.json');continue
            team.load(task);values=[];ds=[]
            with np.load(st.ar.lr.S/'evaluation/cross/q0'/task.replace(':','_')/'scores.npz') as ref:
                old=ref['scores'][0].copy();np.testing.assert_array_equal(ref['indices'],team.pool['action_idx']);np.testing.assert_array_equal(ref['labels'],team.pool['status']>0)
            first=int(old.argmax());failed=team.pool['status'][first]<0;supportpred=None
            with torch.no_grad():
                for start in range(0,1024,32):
                    z,a=team.chunk(start)
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a)
                    fs=[f[None] for f in team.pool['features'][start:start+32]];values.append(scorer(pred,fs).cpu().numpy());ds.append(distances(team.head,pred,fs).cpu().numpy())
                    if first//32*32==start:supportpred=pred[first%32].float().cpu().numpy()
            values=np.concatenate(values);np.testing.assert_array_equal(values,old)
            # All previous pools should preserve canonical float64 simulation action.
            assert 'simulation_action' in team.pool
            source=None
            if failed:
                obs,path=st.strict_observation(team,first);source=str(path)
                st.ar.save_array(out/'support_prediction.npy',supportpred);st.ar.save_array(out/'support_observed.npy',obs['latent'].astype(np.float32))
            st.ar.lr.source().data.save_npz(out/'q0.npz',scores=values,distance=np.concatenate(ds),labels=team.pool['status']>0,indices=team.pool['action_idx'],classifier=team.pool['selection_score'],first=np.array(first))
            write(out/'complete.json',dict(complete=True,plan=sha(RR/'PLAN.json'),task=task,failed=bool(failed),first=first,support_action=team.pool['action'][first].tolist(),source=source,q0_exact=True,hashes={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.npy','.npz']}))
            print('TRANSFER_PACK',task,'failed',failed,flush=True)
    finally:st.ar.lr.close(team)


def evaluate(shard,nshards):
    import torch
    from fast_inference import PreparedScorer
    plan=checked();team=st.team_new();scorer=PreparedScorer(team.head,team.rule);reader=st.ar.lr.source().old.reader_hash(team.model);heads=read(ROOT/'readout/MODELS.json')
    specs=[]
    for seed in [7919,7920]:
        for arm in plan[f'arms_seed{seed}']:specs.append((f'{arm}_s{seed}',ROOT/'models'/f'seed{seed}'/arm,arm))
    specs +=[(arm,ROOT/'calibration_update/models'/arm,arm) for arm in plan['calibration_arms']]
    for _,folder,_ in specs:assert read(folder/'complete.json')['complete']
    try:
        for task in plan['tasks'][shard::nshards]:
            dest=RR/'evaluation'/task.replace(':','_');d=RR/'data'/task.replace(':','_');r=read(d/'complete.json');assert r['complete'] and r['plan']==sha(RR/'PLAN.json')
            if(dest/'result.json').exists():assert read(dest/'result.json')['plan']==sha(RR/'PLAN.json');continue
            for name,h in r['hashes'].items():assert sha(d/name)==h
            team.load(task)
            with np.load(d/'q0.npz') as z:
                base={k:z[k].copy() for k in z.files};np.testing.assert_array_equal(base['indices'],team.pool['action_idx']);np.testing.assert_array_equal(base['labels'],team.pool['status']>0)
            scores={'q0':base['scores'],'classifier':base['classifier']};x=ro.design(base['distance'],base['scores'])
            for arm,h in heads.items():scores[arm]=((x-h['mean'])/h['std'])@h['weight']+h['bias']
            if r['failed']:
                row=dict(support_action=r['support_action'],support_prediction=np.load(d/'support_prediction.npy'),support_observed=np.load(d/'support_observed.npy'))
                with torch.no_grad():
                    for name,folder,arm in specs:
                        receipt=read(folder/'complete.json');assert receipt['checkpoint_sha']==sha(folder/'updater.pt')
                        missing,extra=team.model.load_state_dict(torch.load(folder/'updater.pt',map_location='cpu',weights_only=True),strict=False);assert not extra and all(n.startswith('base.') or n=='initial_tokens' for n in missing)
                        team.input_mode='centered' if arm.endswith('residual') else 'zero';team.model.eval().requires_grad_(False);team.model.checkpoint_blocks=False;state=tr.state_for(team,row);chunks=[]
                        for start in range(0,1024,32):
                            z,a=team.chunk(start)
                            with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a,state.expand(32,-1,-1))
                            chunks.append(scorer(pred,[f[None] for f in team.pool['features'][start:start+32]]).cpu().numpy())
                        scores[name]=np.concatenate(chunks)
            assert reader==st.ar.lr.source().old.reader_hash(team.model)
            st.ar.lr.source().data.save_npz(dest/'scores.npz',**scores,labels=base['labels'],indices=base['indices'],first=base['first'])
            write(dest/'result.json',dict(complete=True,plan=sha(RR/'PLAN.json'),task=task,first_failed=r['failed'],reader_unchanged=True,scorers=list(scores)))
            print('TRANSFER_EVAL',task,len(scores),flush=True)
    finally:st.ar.lr.close(team)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','pack','evaluate']);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=8);a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='pack':pack(a.shard,a.nshards)
    else:evaluate(a.shard,a.nshards)
