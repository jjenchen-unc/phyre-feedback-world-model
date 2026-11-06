"""Structural correctness of five-arm trainability, paired state batching and BPTT."""
import copy
import json
import torch
from common import Predictor
from query_token_ablation import ARMS, Model, pooled


def main():
    torch.set_num_threads(2);torch.manual_seed(73)
    ref=Predictor(8,4,4,16,2,4).double().eval().requires_grad_(False)
    arch=dict(n_tokens=4,d_mem=16,mem_steps=2,mem_grid=2)
    models=[]
    for arm in ARMS:
        m=Model(copy.deepcopy(ref),ref,8,arm,**arch).double()
        assert all(p.requires_grad==(arm!=ARMS[0]) for p in m.base.parameters())
        assert m.initial_tokens.requires_grad==(arm in ARMS[:3])
        assert not any(k.startswith('reference.') for k in m.state_dict())
        assert not any(p is rp for p in m.parameters() for rp in ref.parameters())
        models.append(m)
    a=models[1]
    with torch.no_grad():
        for layer in a.o:layer.weight.normal_(std=.02)
    b=copy.deepcopy(a)
    lengths=[1,2,4,8]
    residual=torch.randn(4,8,4,4,8,dtype=torch.float64)
    actions=torch.rand(4,8,3,dtype=torch.float64)
    z0=torch.randn(4,4,8,dtype=torch.float64);qa=torch.rand(4,3,dtype=torch.float64)
    st=a.initial_state(4)
    for j in range(8):
        alive=torch.tensor([i for i,k in enumerate(lengths) if j<k])
        st=st.index_copy(0,alive,a.update_state(st[alive],residual[alive,j],actions[alive,j]))
    mem,mask=a.state_memory(st,True)
    pred=a(z0,qa,mem,mask);pred.square().mean().backward()
    serial=[]
    for i,k in enumerate(lengths):
        state=b.initial_state()
        for j in range(k):state=b.update_state(state,residual[i:i+1,j],actions[i:i+1,j])
        mem,mask=b.state_memory(state,True)
        pp=b(z0[i:i+1],qa[i:i+1],mem,mask);serial.append(pp)
        (pp.square().mean()/4).backward()
    err=float((pred-torch.cat(serial)).abs().max())
    grads=[]
    for (ka,pa),(kb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert ka==kb
        if pa.requires_grad:
            assert pa.grad is not None and pb.grad is not None
            grads.append(float((pa.grad-pb.grad).abs().max()))
    assert err<1e-10 and max(grads)<1e-10
    assert a.initial_tokens.grad.abs().sum()>0
    assert all(p.grad is None for p in ref.parameters())
    with torch.no_grad():
        # Pooling once into the cache is equivalent to the updater's fixed pooling.
        rr=residual[:1,0];ss=a.initial_state();aa=actions[:1,0]
        assert torch.equal(a.update_state(ss,rr,aa),a.update_state(ss,pooled(rr,2,2),aa))
        for m in models:
            state=m.initial_state(4);mem,mask=m.state_memory(state,False)
            assert torch.equal(m(z0,qa,mem,mask),m.base(z0,qa))
    result={'passed':True,'scope':'CPU float64 small architecture; no runtime/performance claim',
            'batch_vs_serial_prediction_max_error':err,'batch_vs_serial_gradient_max_error':max(grads),
            'depths':lengths,'reference_excluded_from_optimizer_checkpoint':True,
            'reference_gradient_absent':True,'trainability_five_arms':True,'pooling_equivalence':True,
            'empty_uses_current_backbone':True,'full_recurrent_gradient':True}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
