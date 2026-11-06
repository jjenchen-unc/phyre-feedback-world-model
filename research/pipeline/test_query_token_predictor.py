"""Small real-module tests of recurrent state, causal input boundary and gradients."""
import torch
from common import Predictor
from query_token_predictor import QueryTokenPredictor


def main():
    torch.set_num_threads(2);torch.manual_seed(73)
    base=Predictor(8,4,4,16,2,4)
    model=QueryTokenPredictor(base,8,n_tokens=4,d_mem=16,mem_steps=2,mem_grid=2)
    z=torch.randn(2,4,4,8);actions=torch.rand(2,3);state=model.initial_state();original=state.clone()
    s1=model.update_state(state,torch.randn(1,4,4,8),actions[:1]);r2=torch.randn(1,4,4,8)
    s2=model.update_state(s1,r2,actions[1:]);reset=model.update_state(model.initial_state(),r2,actions[1:])
    assert torch.equal(original,state) and not torch.equal(s2,reset)
    s2.square().sum().backward()
    assert model.initial_tokens.grad.abs().sum()>0 and model.update_attention.in_proj_weight.grad.abs().sum()>0
    model.zero_grad();base.eval();model.eval()
    with torch.no_grad():
        for obs in [False,True]:
            mem,mask=model.state_memory(s2,obs)
            assert torch.equal(base(z[:,0],actions),model(z[:,0],actions,mem.expand(2,-1,-1),mask.expand(2,-1)))
        # Nonzero decoder still must leave an unobserved scene at the frozen base.
        for p in model.o:p.weight.normal_(std=.01);p.bias.fill_(.01)
        mem,mask=model.state_memory(model.initial_state(),False)
        assert torch.equal(base(z[:,0],actions),model(z[:,0],actions,mem.expand(2,-1,-1),mask.expand(2,-1)))
    a=torch.rand(1,2,3);ctx=torch.randn(1,2,4,4,8);res=torch.randn_like(ctx);use=torch.ones(1,2,dtype=torch.bool)
    mem,mask=model.encode_memory(ctx,a,torch.zeros(1,2),use,res)
    manual=model.update_state(model.update_state(model.initial_state(),res[:,0],a[:,0]),res[:,1],a[:,1])
    assert torch.equal(mem,manual)
    changed,_=model.encode_memory(ctx+100,a,torch.ones(1,2),use,res)
    assert torch.equal(mem,changed),'Updater must only see residual/action, not full observed latent/outcome'
    with torch.no_grad():
        state_dict={k:v.clone() for k,v in model.state_dict().items()}
        for _ in range(20):s2=model.update_state(s2,r2,actions[:1])
        assert all(torch.equal(v,state_dict[k]) for k,v in model.state_dict().items())
    print('QUERY_TOKEN_CPU_TESTS_PASSED',flush=True)
if __name__=='__main__':main()
