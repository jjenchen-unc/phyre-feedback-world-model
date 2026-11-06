"""Two actual optimizer slots, no serial equivalence checks; short cost pilot."""
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'code'))
import deploy_pool_updater as u
import torch
import torch.distributed as dist


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    rank=int(os.environ['SLURM_PROCID']);world=int(os.environ['SLURM_NTASKS'])
    assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
    dist.init_process_group('gloo',rank=rank,world_size=world,timeout=timedelta(minutes=45))
    team=u.Team('cross','residual',u.data.checked())
    try:
        if rank:team.serve();return
        opt,sched=u.old.optimizer(team.model);reader=u.old.reader_hash(team.model)
        records=[];torch.cuda.reset_peak_memory_stats()
        for slot in range(2):
            torch.cuda.synchronize();start=time.time();opt.zero_grad(set_to_none=True);active=False;events=[]
            for index in range(2*slot,2*slot+2):
                ep=team.collect(index);loss,l1,geometry=team.loss(ep,True,False)
                assert torch.isfinite(loss)
                if loss.requires_grad:(loss/2).backward();active=True
                events.append(team.event(ep,loss,l1,geometry));del loss,ep
            if active:
                norm=torch.nn.utils.clip_grad_norm_([p for p in team.model.parameters() if p.requires_grad],1.)
                assert torch.isfinite(norm);opt.step()
            sched.step();torch.cuda.synchronize()
            records.append({'slot':slot,'seconds':time.time()-start,'events':events})
            print('NATURAL_SLOT',records[-1]['seconds'],[e['used_updates'] for e in events],flush=True)
        assert u.old.reader_hash(team.model)==reader
        mean=sum(r['seconds'] for r in records)/len(records)
        u.data.write(u.R/'COST_PILOT.json',{'passed':True,'provenance':u.provenance(),'panel':'cross','kind':'residual',
              'world_size':world,'records':records,'slot_mean_s':mean,'linear_10000_slots_h':mean*10000/3600,
              'peak_GiB':torch.cuda.max_memory_allocated()/2**30,
              'limits':'Only two natural slots/four scenes; many observations already cached by preflight. Not a reliable total ETA. No serial ranking audits or CPU-backward validation in this timing.'})
        team.request(mode='stop')
    finally:team.truth.close();dist.destroy_process_group()


if __name__=='__main__':main()
