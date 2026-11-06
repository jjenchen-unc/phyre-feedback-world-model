"""Small CPU float64 checks of stage boundaries and gradients through frozen readers."""
import copy
import io
import json
import torch
from common import Predictor
from query_token_staged import Model, is_updater, epoch_indices, fingerprint


def main():
    torch.set_num_threads(2);torch.manual_seed(712)
    base=Predictor(8,4,4,16,2,4).double().eval()
    arch=dict(n_tokens=4,d_mem=16,mem_steps=2,mem_grid=2)
    first=Model(copy.deepcopy(base),8,arch,'q0').double()
    z0=torch.randn(2,4,8,dtype=torch.float64);act=torch.rand(2,3,dtype=torch.float64)
    assert torch.equal(first(z0,act),base(z0,act))
    # Simulate learned reader weights; unlike stage2, stage1 changes the Q0 prediction.
    with torch.no_grad():
        for layer in first.o:layer.weight.normal_(std=.03)
    q0pred=first(z0,act).detach()
    full=Model(copy.deepcopy(base),8,arch,'residual').double()
    full.load_state_dict(first.state_dict());full.reset_updater_identity()
    frozen=fingerprint(full)
    residual=torch.randn(2,4,4,8,dtype=torch.float64)
    state=full.initial_state(2)
    for _ in range(8):state=full.update_state(state,residual,act)
    assert torch.equal(state,full.initial_state(2))
    assert torch.equal(full(z0,act),q0pred)
    # Unmask all gradient routes after testing exact identity initialization.
    with torch.no_grad():
        full.update_attention.out_proj.weight.normal_(std=.02)
        full.update_ff[-1].weight.normal_(std=.02)
    serial=copy.deepcopy(full)
    observed=torch.randn(2,2,4,4,8,dtype=torch.float64)
    actions=torch.rand(2,2,3,dtype=torch.float64)
    st=full.initial_state(2)
    gradients=[]
    for j in range(2):
        r=full.prediction_residual(observed[:,j],actions[:,j],st,j>0)
        if j:assert r.requires_grad
        st=full.update_state(st,r,actions[:,j])
        st.retain_grad();gradients.append(st)
    mem,mask=full.state_memory(st,False)
    assert mask.all(), 'Q0 must remain enabled even at no-history state'
    pred=full(z0,act,mem,mask);pred.square().mean().backward()
    outputs=[]
    for i in range(2):
        st=serial.initial_state()
        for j in range(2):
            r=serial.prediction_residual(observed[i:i+1,j],actions[i:i+1,j],st,j>0)
            st=serial.update_state(st,r,actions[i:i+1,j])
        pp=serial(z0[i:i+1],act[i:i+1],*serial.state_memory(st))
        outputs.append(pp);(pp.square().mean()/2).backward()
    error=float((pred-torch.cat(outputs)).abs().max())
    grad_error=0.
    for (n,p),(nn,pp) in zip(full.named_parameters(),serial.named_parameters()):
        assert n==nn and p.requires_grad==is_updater(n)
        if not p.requires_grad:assert p.grad is None
        else:
            assert p.grad is not None and pp.grad is not None
            grad_error=max(grad_error,float((p.grad-pp.grad).abs().max()))
    assert error<1e-10 and grad_error<1e-10
    assert all(s.grad is not None and s.grad.abs().sum()>0 for s in gradients)
    assert full.residual_in.weight.grad.abs().sum()>0
    opt=torch.optim.AdamW([p for p in full.parameters() if p.requires_grad],lr=1e-3)
    opt.step()
    assert fingerprint(full)==frozen and torch.equal(full(z0,act),q0pred)
    saved=io.BytesIO();torch.save(full.state_dict(),saved);saved.seek(0)
    restored=copy.deepcopy(full);restored.load_state_dict(torch.load(saved,weights_only=True))
    assert torch.equal(restored(z0,act),full(z0,act))
    zero=copy.deepcopy(full);zero.stage='zero'
    assert torch.equal(zero.update_state(zero.initial_state(2),residual,act),zero.update_state(zero.initial_state(2),residual*3,act))
    draws=epoch_indices(52276,0,0)
    assert draws.shape==(6534,8) and len(set(draws.flatten()))==52272 and 8*len(draws)==52272
    print(json.dumps({'passed':True,'batch_prediction_max_error':error,'batch_gradient_max_error':grad_error,
        'active_q0':True,'identity_updater':True,'frozen_reader_input_gradient':True,
        'q0_preserved_after_optimizer_step':True,'full_dynamic_residual_gradient':True,
        'zero_branch_input_invariance':True,'checkpoint_roundtrip':True,'full_epoch_budget':True},indent=2))


if __name__=='__main__':main()
