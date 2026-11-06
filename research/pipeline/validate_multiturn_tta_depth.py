"""Exercise the largest planned TTA learning rate/step count for 19 updates."""
import argparse
import gc
from pathlib import Path
import torch
import validate_multiturn_runtime as v


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    out=v.ROOT/'results/multiturn_validation_20261014'/f'tta_depth_{args.panel}.json'
    assert not out.exists()
    result={'complete':False,'panel':args.panel,'lr':1e-3,'steps':3,'arms':{}}
    source,rows=v.dataset(args.panel)
    row,_,_,_=v.stress_episodes(source,rows)[0];ids=row['candidates']
    z0,actions,features=v.p.initial_data(source,ids)
    observe=lambda j:torch.from_numpy(source.z([ids[j]])).cuda().float()
    with v.Telemetry() as tel:
        def save():result['telemetry']=tel.samples.copy();v.m.write(out,result)
        for arch in ['q0','base']:
            model,probe,head,rule,info=v.load(args.panel,arch)
            snapshot={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
            initial_hash=v.fingerprint(snapshot.items());result['arms'][arch]=[]
            for fast in [False,True]:
                model.load_state_dict(snapshot)
                with v.controlled_backward('cuda',True):
                    records,seconds=v.timed(lambda:v.trajectory(model,head,rule,z0,actions,features,
                        observe,'tta',fast,lr=1e-3,tta_steps=3))
                assert v.fingerprint(model.state_dict().items())==initial_hash
                result['arms'][arch].append({'fast':fast,'seconds':seconds,'events':records,
                    'checkpoint_sha256':info['checkpoint_sha256'],'reset_exact':True});save()
            a,b=[x['events'] for x in result['arms'][arch]]
            assert len(a)==len(b)==20
            for x,y in zip(a,b):
                for key in ['chosen','score_sha256','state_sha256','adaptation']:
                    assert x.get(key)==y.get(key)
            del model,probe,head;gc.collect();torch.cuda.empty_cache()
        result['complete']=True;save()


if __name__=='__main__':main()
