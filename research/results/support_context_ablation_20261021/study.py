"""Support context decomposition with a raw-query-distance information control."""
from pathlib import Path
import importlib.util,json,hashlib,argparse,time
import numpy as np
R=Path(__file__).resolve().parent;PRE=R.parent/'feedback_readout_loop_20261021'
spec=importlib.util.spec_from_file_location('frozen_readout_parent',PRE/'study.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,write,sha=old.read,old.write,old.sha
ARMS=['action_abs','action_rel','action_both','geometry_abs','geometry_rel','geometry_both','absolute_both','relative_both','query_distance','query_distance_action']
BLOCKS={'action_abs':[(77,80)],'action_rel':[(112,115)],'action_both':[(77,80),(112,115)],'geometry_abs':[(40,77)],'geometry_rel':[(80,112)],'geometry_both':[(40,77),(80,112)],'absolute_both':[(40,80)],'relative_both':[(80,115)],'query_distance':[(80,112)],'query_distance_action':[(77,115)]}
def register():
    parent=old.check();assert read(PRE/'FINDINGS.json')['complete'];assert not(R/'PLAN.json').exists()
    p=dict(source_sha=sha(__file__),parent_sources={str(f.relative_to(R.parent)):sha(f) for f in [PRE/'study.py',PRE/'PLAN.json',PRE/'DATA.json',PRE/'NORMALIZATION.json',PRE/'FINDINGS.json']},arms=ARMS,blocks=BLOCKS,seeds=parent['seeds'],panels=parent['panels'],steps=512,lr=.001,weight_decay=.01,delta_penalty=.01,
      design='20freshsmallheads,2seeds x10featureconditions. Allretain original37queryfeatures+queryaction3. Allresidualcoordinateszero. FrozenQ0/readout/candidatepool/firstaction unchanged. Same424queries32positive labels,fullbatch512steps; same131x64x32x1model/init/loss/optimizer asparent.',
      query_distance_control='Replace relative-distance slots80:112 with query raw32distance/256, fit those32mean/std onlyon same424trainingqueries. query_distance_action also retains supportactionandactiondifference. All otherfeatures useoriginalfit-onlynormalization. This controls richerquerytrajectory information available in the relative-distance branch.',
      interpretation='Absolute/action-relative branches contain redundant deterministic information because queryaction isalwaysknown. Geometry-relative adds rawdistance resolution beyond originalsoft-contact37features. Differences canreflect representation/optimization/normalization, not causallynewinformation.',
      shuffle='Cross100 only: usefiveexisting within-template nonidentitypermutations. Swap coherent supportbundle(action,predicted37features,distance32) and recompute query-support differences; keep query/ownfailedactionmask fixed. query_distance branchmustbeinvariant. Diagnostic maybeOOD, notnewdeploymentpolicy.',
      reuse='Replay priorzero(fullsupport) andquery-onlyheads on allvalidation andcross scenes with exact savedscoreagreement. No retrainingthesecontrols. No observedresidualoradditionalworldmodelinference.',
      decision='Reportall20arms/bothseeds andpaireddifferences. Attributevalue tosupport onlyif support-conditioned variants beat matchedrawquerydistancecontrols acrossseeds andproperlysupportedshuffle comparisons. Exploratoryonefold; no bestarmcherry-picking orscaleupfromonecount.',
      limitations=['Onecrossfold/sharedQ0;53trainingfailure scenes and424queries','Allsupportdescriptorsknownbeforeobservingphysics; firstfailureonlypopulation. Doesnotisolatebinaryfailurefeedbackvalue.','Priorstaticreadouthasoneconfirmationlogitoutlier; preservebaseline andanalyzeranking, notaggregateBCE','Featuremaskschangetrainableeffectivecapacity; identicalshapes/initdonotmake informationcontentidentical'])
    write(R/'PLAN.json',p)
def check():
    p=read(R/'PLAN.json');assert sha(__file__)==p['source_sha']
    for f,h in p['parent_sources'].items():assert sha(R.parent/f)==h,f
    old.check();return p
def raw(z,arm,donor=None):
    j=int(z['first']);s=z if donor is None else donor;k=int(s['first']);n=len(z['q']);x=np.zeros((n,131),dtype=np.float64)
    x[:,:37]=z['q'];x[:,37:40]=z['actions'];x[:,40:77]=s['q'][k];x[:,77:80]=s['actions'][k]
    x[:,80:112]=(z['distance']-s['distance'][k])/256;x[:,112:115]=z['actions']-s['actions'][k]
    if arm.startswith('query_distance'):x[:,80:112]=z['distance']/256
    return x
def inputs(z,arm,norm,donor=None):
    v=np.clip((raw(z,arm,donor)-norm['mean'])/norm['std'],-10,10).astype(np.float32);mask=np.zeros(131,dtype=bool);mask[:40]=True
    for lo,hi in BLOCKS[arm]:mask[lo:hi]=True
    v[:,~mask]=0;return v
def prepare():
    import torch
    p=check();_,meta,norm,cache=old.dataset();normalizers={};tests=[]
    for arm in ARMS:
        n=dict(mean=list(norm['mean']),std=list(norm['std']))
        if arm.startswith('query_distance'):
            x=np.concatenate([cache[t]['distance'][cache[t]['query']]/256 for t in p['panels']['fit'] if t in cache]);n['mean'][80:112]=x.mean(0).tolist();n['std'][80:112]=np.maximum(x.std(0),1e-5).tolist()
        normalizers[arm]=n
        t=next(t for t in p['panels']['fit'] if t in cache);z=cache[t];x=inputs(z,arm,n);assert x.shape==(1024,131) and (x[:,115:]==0).all()
        zz={k:v.copy() for k,v in z.items()};zz['labels']=~zz['labels'];np.testing.assert_array_equal(x,inputs(zz,arm,n))
        if arm=='query_distance':
            donor=cache[next(t2 for t2 in cache if t2!=t)];np.testing.assert_array_equal(x,inputs(z,arm,n,donor))
        model=old.network(p['seeds'][0]);delta=model(torch.from_numpy(x).cuda()).squeeze(-1);assert (delta==0).all()
        qi=z['query'];y=torch.from_numpy(z['labels'][qi].astype(np.float32)).cuda();base=torch.from_numpy(z['baseline'][qi]).cuda();loss=torch.nn.functional.binary_cross_entropy_with_logits(base+delta[qi],y);loss.backward();assert model[-1].weight.grad.norm()>0
        tests.append(dict(arm=arm,initial_sha=old.identity(model),output_gradient=float(model[-1].weight.grad.norm())))
    assert len({v['initial_sha'] for v in tests})==1
    write(R/'NORMALIZATIONS.json',normalizers);write(R/'SMOKE.json',dict(passed=True,plan=sha(R/'PLAN.json'),normalizations_sha=sha(R/'NORMALIZATIONS.json'),tests=tests,query_distance_support_invariant=True,labels_not_inputs=True,initialbaseline_exact=True))
    # Exact replay of the two inherited trained controls, all119 activeevalscenes.
    replay={};count=0
    with torch.no_grad():
      for seed in p['seeds']:
       for arm in ['zero','query_only']:
        folder=PRE/'runs'/f'{seed}_{arm}';receipt=read(folder/'complete.json');assert sha(folder/'head.pt')==receipt['checkpoint_sha'];model=old.network(seed);model.load_state_dict(torch.load(folder/'head.pt',map_location='cuda',weights_only=True));model.eval();err=0
        for stage in ['discovery','confirmation','cross100']:
         for task in p['panels'][stage]:
          if task not in cache:continue
          z=cache[task];delta=model(torch.from_numpy(old.inputs(z,z['residual_pc'],arm,norm)).cuda()).squeeze(-1).cpu().numpy();reference=old.load(folder/'evaluation'/stage/task.replace(':','_')/'scores.npz')['actual'];v=z['baseline']+delta;np.testing.assert_array_equal(v,reference);count+=1
        replay[f'{seed}_{arm}']=dict(exact=True,checkpoint_sha=receipt['checkpoint_sha'])
    write(R/'CONTROL_REPLAY.json',dict(passed=True,vectors=count,controls=replay));print('PREPARE_PASS',count,flush=True)
def run(index):
    import torch
    torch.set_num_threads(2);p=check();assert read(R/'SMOKE.json')['passed'] and read(R/'CONTROL_REPLAY.json')['passed'];assert sha(R/'NORMALIZATIONS.json')==read(R/'SMOKE.json')['normalizations_sha']
    _,meta,_,cache=old.dataset();seed=p['seeds'][index//len(ARMS)];arm=ARMS[index%len(ARMS)];norm=read(R/'NORMALIZATIONS.json')[arm];out=R/'runs'/f'{seed}_{arm}';assert not out.exists();net=old.network(seed);initial=old.identity(net)
    xx=[];ys=[];bs=[]
    for task in p['panels']['fit']:
        if task not in cache:continue
        z=cache[task];qi=z['query'];xx.append(inputs(z,arm,norm)[qi]);ys.append(z['labels'][qi]);bs.append(z['baseline'][qi])
    x=torch.from_numpy(np.concatenate(xx)).cuda();y=torch.from_numpy(np.concatenate(ys).astype(np.float32)).cuda();base=torch.from_numpy(np.concatenate(bs)).cuda();assert len(y)==424 and y.sum()==32
    opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['steps']);start=time.time();events=[]
    for step in range(p['steps']):
        opt.zero_grad(set_to_none=True);delta=net(x).squeeze(-1);bce=torch.nn.functional.binary_cross_entropy_with_logits(base+delta,y);loss=bce+p['delta_penalty']*delta.square().mean();assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(g);opt.step();sch.step()
        if (step+1)%32==0:events.append(dict(step=step+1,bce=float(bce.detach()),loss=float(loss.detach()),delta_rms=float(delta.square().mean().sqrt().detach())))
    out.mkdir(parents=True);torch.save(net.cpu().state_dict(),out/'head.pt');np.savez_compressed(out/'weights.npz',**{k:v.numpy() for k,v in net.state_dict().items()});net.cuda().eval();train_seconds=time.time()-start
    with torch.no_grad():
      for stage in ['discovery','confirmation','cross100']:
       for task in p['panels'][stage]:
        if task not in cache:continue
        z=cache[task];a=dict(labels=z['labels'],indices=z['indices'],first=z['first'],q0=z['q0'],static=z['baseline']);a['actual']=z['baseline']+net(torch.from_numpy(inputs(z,arm,norm)).cuda()).squeeze(-1).cpu().numpy()
        if stage=='cross100' and arm!='query_distance':
            for i,mapping in enumerate(meta['shuffle'][stage]):
                donor=cache[mapping[task]];assert mapping[task]!=task
                a[f'shuffle{i}']=z['baseline']+net(torch.from_numpy(inputs(z,arm,norm,donor)).cuda()).squeeze(-1).cpu().numpy()
        d=out/'evaluation'/stage/task.replace(':','_');d.mkdir(parents=True);np.savez_compressed(d/'scores.npz',**a)
    write(out/'events.json',events);write(out/'complete.json',dict(complete=True,plan=sha(R/'PLAN.json'),seed=seed,arm=arm,initial_sha=initial,checkpoint_sha=sha(out/'head.pt'),steps=512,query_labels_sha=hashlib.sha256(np.concatenate(ys).tobytes()).hexdigest(),train_seconds=train_seconds,total_seconds=time.time()-start))
    print('COMPLETE',seed,arm,train_seconds,flush=True)
def integrity():
    p=check();checkpoints={}
    for seed in p['seeds']:
     for arm in ARMS:
        d=R/'runs'/f'{seed}_{arm}';r=read(d/'complete.json');assert r['complete'] and sha(d/'head.pt')==r['checkpoint_sha'];checkpoints[f'{seed}_{arm}']=r['checkpoint_sha']
    write(R/'REMOTE_HASHES.json',dict(checkpoints=checkpoints,files={str(f.relative_to(R)):sha(f) for f in R.rglob('*.npz')}))
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['register','prepare','run','integrity']);q.add_argument('--index',type=int,default=0);a=q.parse_args();run(a.index) if a.mode=='run' else globals()[a.mode]()
