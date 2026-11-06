from study import *
import torch
from train import state_for,terms
from fast_inference import PreparedScorer


def rank(scores,labels,tried):
    mask=np.ones(len(labels),bool);mask[tried]=False;v=scores[mask];y=labels[mask];assert np.isfinite(v).all();order=np.argsort(-v,kind='stable')
    pos=v[y];neg=np.sort(v[~y]);auc=float((np.searchsorted(neg,pos,'left')+np.searchsorted(neg,pos,'right')).sum()/(2*len(pos)*len(neg))) if len(pos) and len(neg) else None
    return {**{f'top{k}':int(y[order[:k]].any()) for k in [1,3,5,8]},'auc':auc}


def run(stage,shard,nshards,seed=None):
    plan=checked();seed=plan['seed'] if seed is None else seed;team=team_new(seed);cal=read(R/'DATA_AND_CALIBRATION.json');scorer=PreparedScorer(team.head,team.rule)
    for arm in ARMS:assert read(R/'models'/f'seed{seed}'/arm/'complete.json')['complete']
    reader=ar.lr.source().old.reader_hash(team.model)
    try:
        for task in plan[stage][shard::nshards]:
            out=R/'evaluation'/f'seed{seed}'/stage/task.replace(':','_')
            if (out/'result.json').exists():assert read(out/'result.json')['plan']==sha(R/'PLAN.json');continue
            row=data(task);team.load(task);labels=team.pool['status']>0;first=row['first'];old=np.load(A/'cache'/task.replace(':','_')/'q0/0/scores.npy')
            score_arrays={'q0':old};records={};geometry={}
            records['q0']=rank(old,labels,[first] if row['failed'] else [])
            if row['failed']:
                for arm in ARMS:
                    ckdir=R/'models'/f'seed{seed}'/arm;rec=read(ckdir/'complete.json');assert rec['checkpoint_sha']==sha(ckdir/'updater.pt')
                    weights=torch.load(ckdir/'updater.pt',map_location='cpu',weights_only=True)
                    missing,extra=team.model.load_state_dict(weights,strict=False);assert not extra and all(k.startswith('base.') or k=='initial_tokens' for k in missing)
                    team.input_mode='centered' if arm.endswith('residual') else 'zero';team.model.eval().requires_grad_(False);team.model.checkpoint_blocks=False
                    with torch.no_grad():
                        state=state_for(team,row);chunks=[]
                        for start in range(0,1024,32):
                            z,a=team.chunk(start)
                            with torch.autocast('cuda',dtype=torch.bfloat16):pred=team.model(z,a,state.expand(32,-1,-1))
                            chunks.append(scorer(pred,[f[None] for f in team.pool['features'][start:start+32]]).cpu().numpy())
                        values=np.concatenate(chunks);score_arrays[arm]=values;records[arm]=rank(values,labels,[first])
                        loss,_,_=terms(team,row,state,cal,grad=False);geometry[arm]={k:float(v) for k,v in loss.items()}
            assert reader==ar.lr.source().old.reader_hash(team.model)
            ar.lr.source().data.save_npz(out/'scores.npz',**score_arrays,labels=labels,indices=team.pool['action_idx'],first=np.array(first))
            write(out/'result.json',dict(complete=True,plan=sha(R/'PLAN.json'),task=task,first_failed=row['failed'],records=records,losses=geometry,reader_unchanged=True))
            print('EVAL',stage,task,records,flush=True)
    finally:ar.lr.close(team)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['discovery','confirmation']);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=8);p.add_argument('--seed',type=int);a=p.parse_args();run(a.stage,a.shard,a.nshards,a.seed)
