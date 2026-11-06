"""Recurrent per-scene query-token state, updated solely from observed residuals."""
import torch
from torch import nn
from torch.nn import functional as F


class QueryTokenPredictor(nn.Module):
    def __init__(self, base, Din, n_tokens=32, d_mem=256, mem_steps=16, mem_grid=10):
        super().__init__()
        self.base=base;self.mem_steps=mem_steps;self.mem_grid=mem_grid;self.n_tokens=n_tokens
        self.base.requires_grad_(False)
        self.initial_tokens=nn.Parameter(torch.randn(1,n_tokens,d_mem)*.02)
        self.residual_in=nn.Linear(Din,d_mem)
        self.action_in=nn.Sequential(nn.Linear(3,d_mem),nn.GELU(),nn.Linear(d_mem,d_mem))
        self.time=nn.Parameter(torch.zeros(1,mem_steps,1,d_mem))
        self.space=nn.Parameter(torch.zeros(1,1,mem_grid*mem_grid,d_mem))
        self.state_norm=nn.LayerNorm(d_mem);self.observation_norm=nn.LayerNorm(d_mem)
        self.update_attention=nn.MultiheadAttention(d_mem,4,batch_first=True,dropout=0.)
        self.ff_norm=nn.LayerNorm(d_mem)
        self.update_ff=nn.Sequential(nn.Linear(d_mem,4*d_mem),nn.GELU(),nn.Linear(4*d_mem,d_mem))
        d=base.inp[1].out_features;self.h=base.blocks[0].sp.h
        self.qn=nn.ModuleList([nn.LayerNorm(d) for _ in base.blocks])
        self.q=nn.ModuleList([nn.Linear(d,d) for _ in base.blocks])
        self.kv=nn.ModuleList([nn.Linear(d_mem,2*d) for _ in base.blocks])
        self.o=nn.ModuleList([nn.Linear(d,d) for _ in base.blocks])
        for p in self.o:nn.init.zeros_(p.weight);nn.init.zeros_(p.bias)

    def initial_state(self, batch=1):
        return self.initial_tokens.expand(batch,-1,-1)

    def update_state(self,state,residual,action):
        """No in-place writes. State retains its autograd history during training."""
        b,t,s,d=residual.shape;g=int(round(s**.5));assert g*g==s and t%self.mem_steps==0
        z=residual[:,::t//self.mem_steps][:,:self.mem_steps]
        z=z.reshape(b*self.mem_steps,g,g,d).permute(0,3,1,2)
        z=F.adaptive_avg_pool2d(z,self.mem_grid).permute(0,2,3,1).reshape(b,self.mem_steps,self.mem_grid**2,d)
        obs=self.residual_in(z)+self.time+self.space+self.action_in(action)[:,None,None]
        obs=self.observation_norm(obs.flatten(1,2))
        update=self.update_attention(self.state_norm(state),obs,obs,need_weights=False)[0]
        state=state+update
        return state+self.update_ff(self.ff_norm(state))

    def state_memory(self,state,observed):
        if isinstance(observed,bool): observed=torch.full((len(state),),observed,device=state.device,dtype=torch.bool)
        return state,observed[:,None].expand(-1,self.n_tokens)

    @torch.no_grad()
    def context_residual(self,ctx_full,ctx_act):
        # Each observation is processed with the same batch-1 base call at train/deploy.
        was=self.base.training;self.base.eval();b,k=ctx_act.shape[:2];out=[]
        for j in range(b):
            row=[]
            for i in range(k):
                z=ctx_full[j,i:i+1]
                pr=self.base(z[:,0].contiguous(),ctx_act[j,i:i+1])
                row.append((z-pr)[0])
            out.append(torch.stack(row))
        self.base.train(was);return torch.stack(out)

    def encode_memory(self,ctx_lat,ctx_act,ctx_out,ctx_mask,ctx_res=None):
        # Compatibility for fixed-context diagnostics. Full latents/outcomes never enter updater.
        assert ctx_res is not None
        state=self.initial_state(len(ctx_act));seen=torch.zeros(len(ctx_act),device=ctx_act.device,dtype=torch.bool)
        for j in range(ctx_act.shape[1]):
            use=ctx_mask[:,j]
            if use.any():
                candidate=self.update_state(state,ctx_res[:,j],ctx_act[:,j])
                state=torch.where(use[:,None,None],candidate,state);seen=seen|use
        return self.state_memory(state,seen)

    def forward(self,z0,action,memory,mask):
        base=self.base
        x=base.inp(z0)[:,None]+base.time+base.space+base.act(action)[:,None,None]
        b,t,s,d=x.shape;gate=mask.any(1).to(x.dtype)[:,None,None,None]
        for j,block in enumerate(base.blocks):
            x=block(x)
            q=self.q[j](self.qn[j](x)).reshape(b,t*s,self.h,d//self.h).transpose(1,2)
            k,v=self.kv[j](memory).reshape(b,memory.shape[1],2,self.h,d//self.h).permute(2,0,3,1,4)
            att=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:])
            x=x+gate*self.o[j](att.transpose(1,2).reshape(b,t,s,d))
        pred=z0[:,None]+base.out(x)
        return torch.cat([z0[:,None],pred[:,1:]],1)
