"""Closed-loop interventions on a frozen updater; original controls stay immutable."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
P=Path('/path/to/phyre-unc');SOURCE=P/'results/q0_first_feedback_20261018'
R=P/'results/memory_overnight_policy_20261019'
sys.path[:0]=[str(SOURCE/'code'),str(SOURCE/'sunflower_backend'),str(P/'pipeline')]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def component(z,variant):
    if variant=='mean_only':return z.mean((1,2),keepdim=True).expand_as(z)
    if variant=='centered':return z-z.mean((1,2),keepdim=True)
    return z

def intervention(model,state,residual,action,variant):
    import torch
    if variant in ['mean_only','centered']:
        hook=model.residual_in.register_forward_pre_hook(lambda module,args:(component(args[0],variant),))
        try:return model.update_state(state,residual,action)
        finally:hook.remove()
    real=model.update_state(state,residual,action)
    if variant=='real':return real
    if variant.startswith('step_'):return state+float(variant.split('_')[1])*(real-state)
    if variant=='feedback_only':
        zero=model.update_state(state,torch.zeros_like(residual),action)
        return state+(real-zero)
    raise ValueError(variant)

def register(variants):
    source=read(SOURCE/'PLAN.json')
    allowed={'real','mean_only','centered','step_0.25','step_0.5','feedback_only'}
    assert set(variants)<=allowed and len(variants)==len(set(variants))
    plan={'source_plan_sha256':sha(SOURCE/'PLAN.json'),'code_sha256':sha(__file__),
        'variants':variants,'validation_tasks':{p:read(P/'results/residual_mean_step_diagnostic_20261019/PLAN.json')['tasks'][p][0] for p in ['cross','within']},'tasks':{p:x['tasks'] for p,x in source['panels'].items()},'budget':9,
        'checkpoints':{p:read(SOURCE/'models'/p/'residual'/'complete.json') for p in ['cross','within']},
        'selection':'All original100testscenes/protocol; no outcome filtering',
        'observation_policy':'Reuse canonical immutable original cache if present; write new observations to this experiment only using canonical A6000 RPC encoder',
        'controls':'Reuse original Q0/residual/zero/tta outcomes; firstscores must exactly reproduce original residual for every task',
        'limits':['Same seed and folds; intervention only, no retraining','Policy uses observed actions only; untried labels are evaluation outcomes, not ranking inputs','State trajectories diverge naturally after first action','Scale/representation interventions may be OOD']}
    if (R/'PLAN.json').exists():assert read(R/'PLAN.json')==plan
    else:write(R/'PLAN.json',plan)
    print('REGISTERED',variants,flush=True)

def run(panel,variant,shard,nshards):
    import numpy as np,torch
    import selector_feedback_run as src
    from runtime import IsolatedEncoder
    from fast_inference import CandidateRanker
    torch.set_num_threads(2);src.data.EncoderProxy=IsolatedEncoder
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    plan=read(R/'PLAN.json');assert sha(__file__)==plan['code_sha256'] and sha(SOURCE/'PLAN.json')==plan['source_plan_sha256']
    assert variant in plan['variants'];ck=plan['checkpoints'][panel];assert sha(ck['checkpoint'])==ck['checkpoint_sha256']
    class OwnTruth(src.TimedTruthCache):
        def __init__(self,panel):
            super().__init__(panel);self.parent_root=self.root;self.root=R/'observations'/panel
        def get(self,task,local,pool):
            key=f'{task.replace(":","_")}_{int(pool["action_idx"][local]):06d}'
            path=self.parent_root/str(int(hashlib.sha256(key.encode()).hexdigest(),16)%32)/(key+'.npz')
            if path.exists():
                start=time.time()
                with np.load(path) as z:
                    assert str(z['plan'])==sha(src.data.SOURCE_R/'PLAN.json') and str(z['task'])==task
                    assert int(z['status'])==int(pool['status'][local])
                    np.testing.assert_array_equal(z['action'],pool['simulation_action'][local])
                    value={'local':local,'latent':z['latent'].copy(),'features':z['features'].copy(),'status':int(z['status']),
                        'action_idx':int(pool['action_idx'][local]),'cache_hit':True,'total_observation_s':time.time()-start,
                        'observation_encode_s':0.,'cache_source':'original_readonly'}
                return value
            value=super().get(task,local,pool);value['cache_source']='new_experiment';return value
    team=src.Team(panel,'residual',src.data.checked());team.truth.close();team.truth=OwnTruth(panel)
    team.model.load_state_dict(torch.load(ck['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
    team.model.eval().requires_grad_(False);initial=src.old.ident(team.model)
    prov={'plan_sha256':sha(R/'PLAN.json'),'checkpoint_sha256':ck['checkpoint_sha256']}
    try:
        for task in plan['tasks'][panel][shard::nshards]:
            if variant=='real' and task!=plan['validation_tasks'][panel]:continue
            dest=R/'evaluation'/panel/variant/task.replace(':','_')
            if (dest/'result.json').exists():assert read(dest/'result.json')['provenance']==prov;continue
            team.load(task);pool=team.pool;z0=torch.from_numpy(pool['z0'].copy()).cuda().float();actions=torch.from_numpy(pool['action']).cuda()
            ranker=CandidateRanker(team.model,team.head,team.rule,z0,actions,[f[None] for f in pool['features']])
            state=team.model.initial_state().detach().clone();tried=np.zeros(1024,bool);selected=[];history=[];events=[];success=None;began=time.time()
            source=SOURCE/'evaluation'/panel/'residual'/task.replace(':','_')/'scores.npz'
            with np.load(source) as z:
                expected_history=z['scores'].copy();expected_first=expected_history[0];expected_indices=z['indices'].copy();expected_labels=z['labels'].copy()
            np.testing.assert_array_equal(expected_indices,pool['action_idx']);np.testing.assert_array_equal(expected_labels,pool['status']>0)
            with torch.no_grad():
                for turn in range(plan['budget']):
                    ranked,rank_s=src.timed(lambda:ranker.rank(state,tried));values=ranked.scores
                    if turn==0:np.testing.assert_array_equal(values,expected_first)
                    if variant=='real':np.testing.assert_array_equal(values,expected_history[turn])
                    j=src.pools.choose_trial(values,tried,turn);assert not tried[j];tried[j]=True;selected.append(j);history.append(values)
                    event={'turn':turn+1,'local':j,'action_idx':int(pool['action_idx'][j]),'solved':bool(pool['status'][j]>0),'rank_s':rank_s,'update_s':0.,'observation_s':0.};events.append(event)
                    if event['solved']:success=turn+1;break
                    if turn+1==plan['budget']:break
                    obs,event['observation_s']=src.timed(lambda:team.observations([j])[j])
                    event.update(cache_hit=obs['cache_hit'],cache_source=obs['cache_source'],observation_encode_s=obs['observation_encode_s'])
                    def update():
                        pred=team.predict(state,j)[j%32:j%32+1]
                        residual=torch.from_numpy(obs['latent'][None]).cuda().float()-pred
                        with torch.autocast('cuda',dtype=torch.bfloat16):return intervention(team.model,state,residual,actions[j:j+1],variant)
                    state,event['update_s']=src.timed(update)
            assert src.old.ident(team.model)==initial
            for turn,j in enumerate(selected):
                scores=history[turn].copy();scores[selected[:turn]]=-np.inf;assert int(scores.argmax())==j
            src.data.save_npz(dest/'scores.npz',scores=np.stack(history),selected=np.array(selected),indices=pool['action_idx'],labels=pool['status']>0)
            write(dest/'result.json',{'complete':True,'provenance':prov,'panel':panel,'variant':variant,'task':task,'first_success':success,
                'events':events,'first_scores_exact':True,'model_unchanged':True,'seconds':time.time()-began,'gpu':torch.cuda.get_device_name(0)})
            print('EVAL',panel,variant,task,success,round(time.time()-began,1),flush=True)
    finally:team.truth.close()
    write(R/'shards'/f'{panel}_{variant}_{shard}.json',{'complete':True,'provenance':prov})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','run']);ap.add_argument('--variants',default='');ap.add_argument('--panel');ap.add_argument('--variant');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=1);a=ap.parse_args()
    register(a.variants.split(',')) if a.mode=='register' else run(a.panel,a.variant,a.shard,a.nshards)
