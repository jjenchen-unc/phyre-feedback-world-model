"""Memory prefix tokens, residual cross-attention, causal rollout and full BPTT."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class PrefixPredictor(nn.Module):
    def __init__(self, base, din, n_tokens=32, mem_steps=16, mem_grid=10,
                 stage='joint', zero_residual=False, checkpoint_blocks=True):
        super().__init__()
        assert stage in ['initial','joint','updater','frozen_backbone']
        self.base=base;self.n_tokens=n_tokens;self.mem_steps=mem_steps;self.mem_grid=mem_grid
        self.d=base.inp[1].out_features;self.zero_residual=zero_residual
        self.checkpoint_blocks=checkpoint_blocks
        self.initial_tokens=nn.Parameter(torch.randn(1,n_tokens,self.d)*.02)
        self.residual_in=nn.Linear(din,self.d)
        self.action_in=nn.Sequential(nn.Linear(3,self.d),nn.GELU(),nn.Linear(self.d,self.d))
        self.update_time=nn.Parameter(torch.zeros(1,mem_steps,1,self.d))
        self.update_space=nn.Parameter(torch.zeros(1,1,mem_grid**2,self.d))
        self.state_norm=nn.LayerNorm(self.d);self.observation_norm=nn.LayerNorm(self.d)
        self.update_attention=nn.MultiheadAttention(self.d,base.blocks[0].sp.h,batch_first=True,dropout=0.)
        self.ff_norm=nn.LayerNorm(self.d)
        self.update_ff=nn.Sequential(nn.Linear(self.d,4*self.d),nn.GELU(),nn.Linear(4*self.d,self.d))
        self.reset_updater_identity();self.set_stage(stage)

    def set_stage(self,stage):
        self.stage=stage
        for name,p in self.named_parameters():
            reader=name.startswith('base.') or name=='initial_tokens'
            p.requires_grad_((reader if stage=='initial' else not reader if stage=='updater'
                              else not name.startswith('base.') if stage=='frozen_backbone' else True))

    def reset_updater_identity(self):
        for layer in [self.update_attention.out_proj,self.update_ff[-1]]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def initial_state(self,batch=1):return self.initial_tokens.expand(batch,-1,-1)

    def forward(self,z0,actions,state=None):
        if state is None:state=self.initial_state(len(z0))
        assert state.shape==(len(z0),self.n_tokens,self.d)
        base=self.base
        x=base.inp(z0)[:,None]+base.time+base.space+base.act(actions)[:,None,None]
        # A shared episode prefix is broadcast across time slots. The backbone's
        # original blocks jointly self-attend over [prefix, scene] spatial slots,
        # then across time for every slot, including prefix slots.
        prefix=state[:,None].expand(-1,x.shape[1],-1,-1)+base.time
        x=torch.cat([prefix,x],dim=2)
        for block in base.blocks:
            x=checkpoint(block,x,use_reentrant=False) if self.checkpoint_blocks and torch.is_grad_enabled() and x.requires_grad else block(x)
        x=x[:,:,self.n_tokens:]
        pred=z0[:,None]+base.out(x)
        return torch.cat([z0[:,None],pred[:,1:]],dim=1)

    def update_state(self,state,residual,action):
        if self.zero_residual:residual=torch.zeros_like(residual)
        b,t,s,din=residual.shape;grid=int(round(s**.5))
        assert grid*grid==s and t%self.mem_steps==0
        z=residual[:,::t//self.mem_steps]
        z=z.reshape(b*self.mem_steps,grid,grid,din).permute(0,3,1,2)
        z=F.adaptive_avg_pool2d(z,self.mem_grid).permute(0,2,3,1).reshape(b,self.mem_steps,self.mem_grid**2,din)
        obs=self.residual_in(z)+self.update_time+self.update_space+self.action_in(action)[:,None,None]
        obs=self.observation_norm(obs.flatten(1,2))
        u=self.update_attention(self.state_norm(state),obs,obs,need_weights=False)[0]
        state=state+u
        return state+self.update_ff(self.ff_norm(state))


@torch.no_grad()
def choose_action(model,state,z0,actions,initial_features,score_fn,tried,batch_size=32):
    """No labels or future trajectory argument; stable original-order tie break."""
    assert len(z0)==len(actions)==len(initial_features)==len(tried)
    values=[];best=-float('inf');choice=None;chosen_prediction=None
    for start in range(0,len(actions),batch_size):
        end=min(start+batch_size,len(actions))
        pred=model(z0[start:end],actions[start:end],state.expand(end-start,-1,-1))
        scores=score_fn(pred,initial_features[start:end]).detach()
        assert scores.shape==(end-start,) and torch.isfinite(scores).all()
        values.append(scores)
        eligible=scores.masked_fill(tried[start:end],-float('inf'))
        j=int(torch.argmax(eligible));v=float(eligible[j])
        if v>best:best=v;choice=start+j;chosen_prediction=pred[j:j+1].clone()
    return choice,torch.cat(values),chosen_prediction


@torch.no_grad()
def policy_rollout(model,z0,actions,initial_features,score_fn,observe,budget=9,batch_size=32):
    """On-current-policy collection, frozen weights for this episode.

    observe receives ONLY the chosen local action ID. The caller may retrieve
    cached truth for that action. Failed actions become supports; stop at success.
    Saved scores and predictions are diagnostic only, never stale training residuals.
    """
    state=model.initial_state().detach().clone()
    tried=torch.zeros(len(actions),dtype=torch.bool,device=actions.device)
    trace=[];supports=[]
    for attempt in range(1,min(budget,len(actions))+1):
        chosen,scores,pred=choose_action(model,state,z0,actions,initial_features,score_fn,tried,batch_size)
        if chosen is None:break
        assert not tried[chosen];tried[chosen]=True
        solved,truth=observe(chosen)
        trace.append({'attempt':attempt,'local':chosen,'solved':bool(solved),
                      'scores':scores.cpu(),'state':state.cpu().clone()})
        if solved or attempt==budget or tried.all():break
        residual=truth-pred
        supports.append(chosen)
        state=model.update_state(state,residual,actions[chosen:chosen+1])
    return trace,supports


def replay_state(model,z0,actions,supports,truth_for,batch_size=32):
    """Full BPTT replay at unchanged weights; recompute each current residual.

    Recompute the candidate chunk that contained the chosen action, matching the
    policy forward shape. No cached residuals or detached input state are used.
    """
    state=model.initial_state()
    for chosen in supports:
        start=chosen//batch_size*batch_size;end=min(start+batch_size,len(actions))
        pred=model(z0[start:end],actions[start:end],state.expand(end-start,-1,-1))
        residual=truth_for(chosen)-pred[chosen-start:chosen-start+1]
        state=model.update_state(state,residual,actions[chosen:chosen+1])
    return state
