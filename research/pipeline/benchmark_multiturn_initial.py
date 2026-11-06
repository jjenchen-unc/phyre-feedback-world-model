"""Bounded full-data initial-model optimizer throughput and checkpoint I/O."""
import argparse,gc,time
from pathlib import Path
import torch
import matched_base_q0 as b
from validate_multiturn_runtime import timed,Telemetry


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();assert not args.output.exists();torch.set_num_threads(4)
    plan=b.m.read(b.ROOT/'PLAN.json');spec=plan['panels'][args.panel];weight=0. if args.panel=='cross' else .03
    result={'complete':False,'panel':args.panel,'weight':weight,'scope':'256 actual optimizer steps per initial architecture; not full training','arms':{}}
    with Telemetry() as tel:
        def save():result['telemetry']=tel.samples.copy();b.m.write(args.output,result)
        start=time.perf_counter();source=b.p.f.Pack(spec['train_pack']);dev=b.p.f.Pack(spec['dev_pack'])
        result['pack_load_s']=time.perf_counter()-start
        draws=b.p.old.epoch_indices(spec['n_train'],0,0)
        for arch in ['base','q0']:
            model,probe,head,rule,_=b.models(plan,args.panel,arch,0)
            model.train();params,opt,sched=b.optimizer(model,spec['updates'])
            arm={'steps':[]};result['arms'][arch]=arm;save();torch.cuda.reset_peak_memory_stats()
            for i in range(256):
                def step():
                    opt.zero_grad(set_to_none=True)
                    l,g=b.terms(model,probe,source,draws[i].tolist());loss=l+weight*g
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();sched.step()
                    return float(loss.detach())
                loss,seconds=timed(step);arm['steps'].append({'seconds':seconds,'warmup':i<16,'loss':loss})
                if i%32==31:save();print('INITIAL',args.panel,arch,i+1,seconds,flush=True)
            _,arm['dev400_s']=timed(lambda:b.diagnostics(model,probe,dev,spec['dev_ids']))
            path=args.output.parent/f'initial_io_{args.panel}_{arch}.pt'
            _,arm['checkpoint_write_s']=timed(lambda:b.atomic_save(path,{'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict()}))
            arm['checkpoint_bytes']=path.stat().st_size;path.unlink()
            arm['peak_GiB']=torch.cuda.max_memory_allocated()/2**30
            save();del model,probe,head,opt,sched,params;gc.collect();torch.cuda.empty_cache()
        result['complete']=True;save()


if __name__=='__main__':main()
