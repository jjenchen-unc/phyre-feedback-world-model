"""Frozen-world-model feedback readout pilot. Cached observations only."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parent;P=R.parents[1];OLD=P/'results/success_objective_loop_20261021'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def design(d,s):
    soft=1/(1+np.exp(np.clip(d.astype(float)-4,-700,700)));run=np.zeros(len(d));best=run.copy()
    for v in (d<4).T:run=(run+1)*v;best=np.maximum(best,run)
    return np.column_stack([soft,s,d.min(1)/256,d.mean(1)/256,d.max(1)/256,best/32])
def register():
    assert read(OLD/'FINAL_FINDINGS.json')['complete']
    p=read(OLD/'PLAN.json');panels={k:p[k] for k in ['fit','discovery','confirmation']};panels['cross100']=read(OLD/'cross_transfer/EXTENSION_PLAN.json')['all_tasks']
    plan=dict(source_sha=sha(__file__),parents={str(q.relative_to(P)):sha(q) for q in [OLD/'PLAN.json',OLD/'readout/MODELS.json',OLD/'cross_transfer/EXTENSION_PLAN.json',OLD/'INTEGRITY_AUDIT.json']},panels=panels,
      seeds=[8419,8420],arms=['real','zero','query_only'],steps=512,lr=.001,weight_decay=.01,delta_penalty=.01,
      architecture='Frozen query8 logistic readout + zero-output-initialized MLP(131,64,32,1), GELU. Query37features/action3; support37features/action3; query-support distance32/actiondiff3; residual16PC. Support failure implicit: only firstQ0failure population.',
      residual='Observed minus current frozenQ0 prediction, centered perchannel overtime/spatial. Pool32x14x14x128 into8x2x2x128 meanandRMS; uncenteredSVD fit53supports only, retain16PC. No cross/test fitting. Zero residualmaps toexactzeroPC.',
      data='Same53fitfailure scenes/424uniformqueries/32positives as previousloop. FixedQ0first andclassifier1024pool. No label-basedqueries, newobservations, simulations, encoder orpredictor inference.',
      optimizer='Full424querybatch BCEWithLogits + .01mean(delta^2); AdamW.001 wd.01 cosine512,clip1. Sameweights andsteps acrossconditionswithinseed. Finalcheckpointonly; noeval-basedselection.',
      normalization='Fitqueriesonly featuremean/std floor1e-5; residualmeanfixedzero. ClipnormalizedMLPinputs to[-10,10]; frozenbaselineunclipped. Queryonlyzeros75support+16residualcoordinates after scaling. Zeroconditionzerosonly16residualcoordinates.',
      shuffle='Five seeded cyclic nonidentitypermutations within template/stage amongfirstfailures; retain ownsupportaction andpredictiongeometry; swapresidualPC only. Permutation breaks association, maybeOOD; nottreatedasfairtrainedmethod.',
      gate='Expand onlyif BOTHseeds conditionalcross top1 strictlybeatsstaticreadout,trainedzeroandqueryonly; no top5regressionvsstatic, and realtop1beatsmean5shuffles. This is exploratory gate, notsignificance proof.',
      limits=['Onecrossfold/sharedQ0; thispilotdoesnotestablishmultiturn/withinperformance','Compressedresidual canloseinformation; negative resultcannotruleoutfullresidualhead','Small424querydataset; no attributingfailuretosizealone','Staticreadoutpretrainedonsame424labels, allnewarmsreceiveexactsameinitialreadout','Noadditionalfoldunlessfeedbackgatepasses'])
    assert not(R/'PLAN.json').exists();write(R/'PLAN.json',plan)
def check():
    p=read(R/'PLAN.json');assert p['source_sha']==sha(__file__)
    for path,h in p['parents'].items():assert sha(P/path)==h,path
    return p
def pooled(observed,predicted):
    r=observed.astype(np.float64)-predicted.astype(np.float64);assert r.shape==(32,196,128)
    r-=r.mean((0,1),keepdims=True);v=r.reshape(8,4,2,7,2,7,128)
    mean=v.mean((1,3,5));rms=np.sqrt((v*v).mean((1,3,5)))
    return np.concatenate([mean.ravel(),rms.ravel()])
def pack():
    p=check();h=read(OLD/'readout/MODELS.json')['query8_readout'];records={};cache={}
    for stage,tasks in p['panels'].items():
      for task in tasks:
        name=task.replace(':','_');cross=stage=='cross100';d=OLD/('cross_transfer/data' if cross else 'data')/name;m=read(d/'complete.json');assert m['complete']
        if not m['failed']:records[task]=dict(stage=stage,failed=False,first=m['first']);continue
        for fn,hsh in m.get('hashes',{}).items():assert sha(d/fn)==hsh,(task,fn)
        if cross:
            z=load(d/'q0.npz');scores=z['scores'];pool=P/'results/selector_feedback_pilot_20261016/pools/cross'/name/'pool.npz'
        else:
            src=OLD/'readout/data'/name;receipt=read(src/'complete.json');assert receipt['sha256']==sha(src/'features.npz');z=load(src/'features.npz');scores=z['old_score'];pool=P/'results/current_probe_adaptation_20261020/pools'/name/'pool.npz'
        pp=load(pool);np.testing.assert_array_equal(pp['action_idx'],z['indices']);np.testing.assert_array_equal(pp['status']>0,z['labels']);assert (pp['status']!=0).all()
        first=int(z['first']);assert first==m['first']==int(scores.argmax()) and not z['labels'][first]
        action=pp['action'];np.testing.assert_array_equal(action[first],np.asarray(m['support_action'],dtype=action.dtype))
        q=design(z['distance'],scores);base=((q-h['mean'])/h['std'])@h['weight']+h['bias']
        ref=load(OLD/('cross_transfer/evaluation' if cross else 'readout/evaluation/'+stage)/name/'scores.npz')
        assert np.max(abs(base-ref['query8_readout']))<1e-10
        r=pooled(np.load(d/'support_observed.npy'),np.load(d/'support_prediction.npy'))
        query=np.array(m.get('query',[]),dtype=int);assert first not in query
        if len(query):np.testing.assert_array_equal(z['labels'][query],m['labels'])
        cache[task]=dict(q=q,actions=action,distance=z['distance'],residual=r,labels=z['labels'],indices=z['indices'],first=np.array(first),query=query,baseline=base,q0=scores)
        records[task]=dict(stage=stage,failed=True,first=first,source_sha={str(f.relative_to(P)):sha(f) for f in [d/'complete.json',pool]},pooled_norm=float(np.linalg.norm(r)))
    fit=[t for t in p['panels']['fit'] if t in cache];x=np.stack([cache[t]['residual'] for t in fit]);u,s,vh=np.linalg.svd(x,full_matrices=False);components=vh[:16].T
    # Zero residual is zero projection, unlike mean-centered PCA.
    assert np.array_equal(np.zeros(x.shape[1])@components,np.zeros(16))
    retained={}
    for task,z in cache.items():
        raw=z.pop('residual');pc=raw@components;z['residual_pc']=pc
        retained[task]=float(np.square(pc).sum()/max(np.square(raw).sum(),1e-20))
        out=R/'data'/task.replace(':','_');out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'features.npz',**z);records[task]['data_sha']=sha(out/'features.npz')
    np.savez_compressed(R/'RESIDUAL_PROJECTION.npz',components=components,singular_values=s)
    # Normalization fitted only on the same424 labeledtrainingqueries.
    rows=[]
    for task in fit:
        z=cache[task];rows.append(raw_input(z,z['residual_pc'])[z['query']])
    xx=np.concatenate(rows);mean=xx.mean(0);std=np.maximum(xx.std(0),1e-5);mean[-16:]=0
    write(R/'NORMALIZATION.json',dict(mean=mean.tolist(),std=std.tolist(),train_examples=len(xx)))
    shuffled={}
    for stage,tasks in p['panels'].items():
        groups={}
        for t in tasks:
            if t in cache:groups.setdefault(t[:5],[]).append(t)
        shuffled[stage]=[]
        for seed in range(5):
            mapping={}
            for template,ts in groups.items():
                if len(ts)<2: mapping.update({t:t for t in ts});continue
                rng=np.random.default_rng([20261021,seed,int(template)]);order=list(rng.permutation(sorted(ts)));offset=int(rng.integers(1,len(ts)))
                mapping.update({t:order[(i+offset)%len(ts)] for i,t in enumerate(order)})
            shuffled[stage].append(mapping)
    write(R/'DATA.json',dict(complete=True,plan=sha(R/'PLAN.json'),records=records,shuffle=shuffled,projection_sha=sha(R/'RESIDUAL_PROJECTION.npz'),normalization_sha=sha(R/'NORMALIZATION.json'),fit_projection_energy=float((s[:16]**2).sum()/(s**2).sum()),retained_by_scene=retained,
       counts={stage:sum(t in cache for t in ts) for stage,ts in p['panels'].items()},training_queries=len(xx),training_positives=int(sum(cache[t]['labels'][cache[t]['query']].sum() for t in fit))))
    print('PACK_COMPLETE',len(cache),'fit energy',float((s[:16]**2).sum()/(s**2).sum()),flush=True)
def raw_input(z,pc):
    j=int(z['first']);n=len(z['q']);support=np.broadcast_to(z['q'][j],(n,37));a=z['actions'];sa=np.broadcast_to(a[j],a.shape)
    return np.column_stack([z['q'],a,support,sa,(z['distance']-z['distance'][j])/256,a-a[j],np.broadcast_to(pc,(n,16))])
def inputs(z,pc,condition,norm):
    v=(raw_input(z,pc)-norm['mean'])/norm['std'];v=np.clip(v,-10,10)
    if condition=='zero':v[:,-16:]=0
    if condition=='query_only':v[:,40:]=0
    return v.astype(np.float32)
def network(seed):
    import torch
    torch.manual_seed(seed);net=torch.nn.Sequential(torch.nn.Linear(131,64),torch.nn.GELU(),torch.nn.Linear(64,32),torch.nn.GELU(),torch.nn.Linear(32,1))
    torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias);return net.cuda()
def identity(net):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
def dataset():
    p=check();meta=read(R/'DATA.json');assert meta['complete'] and meta['plan']==sha(R/'PLAN.json');norm=read(R/'NORMALIZATION.json');assert meta['normalization_sha']==sha(R/'NORMALIZATION.json') and meta['projection_sha']==sha(R/'RESIDUAL_PROJECTION.npz')
    cache={}
    for t,r in meta['records'].items():
        if not r['failed']:continue
        path=R/'data'/t.replace(':','_')/'features.npz';assert sha(path)==r['data_sha'];cache[t]=load(path)
    return p,meta,norm,cache
def smoke():
    import torch
    p,meta,norm,cache=dataset();t=next(t for t in p['panels']['fit'] if t in cache);z=cache[t];net=network(p['seeds'][0]);x=inputs(z,z['residual_pc'],'real',norm);tensor=torch.from_numpy(x).cuda()
    assert x.shape==(1024,131);assert (inputs(z,z['residual_pc'],'zero',norm)[:,-16:]==0).all();assert (inputs(z,z['residual_pc'],'query_only',norm)[:,40:]==0).all()
    zz={k:v.copy() for k,v in z.items()};zz['labels']=~zz['labels'];np.testing.assert_array_equal(inputs(zz,z['residual_pc'],'real',norm),x)
    delta=net(tensor).squeeze(-1);assert (delta==0).all();np.testing.assert_array_equal(z['baseline']+delta.detach().cpu().numpy(),z['baseline'])
    y=torch.from_numpy(z['labels'][z['query']].astype(np.float32)).cuda();base=torch.from_numpy(z['baseline'][z['query']]).cuda();loss=torch.nn.functional.binary_cross_entropy_with_logits(base+delta[z['query']],y);loss.backward();assert net[-1].weight.grad.norm()>0
    assert np.array_equal(inputs(z,z['residual_pc'],'zero',norm),inputs(z,z['residual_pc']*17,'zero',norm))
    write(R/'SMOKE.json',dict(passed=True,initial_sha=identity(net),initial_baseline_exact=True,zero_and_query_only_invariants=True,labels_not_inputs=True,nonzero_output_gradient=True,parameter_count=sum(v.numel() for v in net.parameters())))
    print('SMOKE_PASS',flush=True)
def run(index):
    import torch
    torch.set_num_threads(2);p,meta,norm,cache=dataset();assert read(R/'SMOKE.json')['passed'];seed=p['seeds'][index//3];arm=p['arms'][index%3];out=R/'runs'/f'{seed}_{arm}';assert not out.exists(), 'Preserve existing run; inspect before retry'
    net=network(seed);initial=identity(net);xx=[];yy=[];bb=[]
    for t in p['panels']['fit']:
        if t not in cache:continue
        z=cache[t];qi=z['query'];xx.append(inputs(z,z['residual_pc'],arm,norm)[qi]);yy.append(z['labels'][qi]);bb.append(z['baseline'][qi])
    x=torch.from_numpy(np.concatenate(xx)).cuda();y=torch.from_numpy(np.concatenate(yy).astype(np.float32)).cuda();base=torch.from_numpy(np.concatenate(bb)).cuda();assert len(y)==424 and int(y.sum())==32
    opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['steps']);events=[];start=time.time()
    for step in range(p['steps']):
        opt.zero_grad(set_to_none=True);delta=net(x).squeeze(-1);bce=torch.nn.functional.binary_cross_entropy_with_logits(base+delta,y);loss=bce+p['delta_penalty']*delta.square().mean();assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(gn);opt.step();sch.step()
        if (step+1)%32==0:events.append(dict(step=step+1,bce=float(bce.detach()),loss=float(loss.detach()),delta_rms=float(delta.square().mean().sqrt().detach())))
    out.mkdir(parents=True,exist_ok=True);torch.save(net.cpu().state_dict(),out/'head.pt');net.cuda().eval();train_seconds=time.time()-start
    with torch.no_grad():
      for stage,tasks in p['panels'].items():
        if stage=='fit':continue
        for task in tasks:
            if task not in cache:continue
            z=cache[task];a=dict(labels=z['labels'],indices=z['indices'],first=z['first'],q0=z['q0'],static=z['baseline'])
            a['actual']=z['baseline']+net(torch.from_numpy(inputs(z,z['residual_pc'],arm,norm)).cuda()).squeeze(-1).cpu().numpy()
            if arm=='real':
                for i,mapping in enumerate(meta['shuffle'][stage]):
                    donor=cache[mapping[task]]['residual_pc'];a[f'shuffle{i}']=z['baseline']+net(torch.from_numpy(inputs(z,donor,arm,norm)).cuda()).squeeze(-1).cpu().numpy()
            folder=out/'evaluation'/stage/task.replace(':','_');folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'scores.npz',**a)
    write(out/'events.json',events);write(out/'complete.json',dict(complete=True,plan=sha(R/'PLAN.json'),data=sha(R/'DATA.json'),seed=seed,arm=arm,initial_sha=initial,checkpoint_sha=sha(out/'head.pt'),train_seconds=train_seconds,total_seconds=time.time()-start,steps=p['steps'],query_labels_sha=hashlib.sha256(np.concatenate(yy).tobytes()).hexdigest()))
    print('RUN_COMPLETE',seed,arm,train_seconds,events[-1],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','pack','smoke','run']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.mode=='run':run(a.index)
    else:globals()[a.mode]()
