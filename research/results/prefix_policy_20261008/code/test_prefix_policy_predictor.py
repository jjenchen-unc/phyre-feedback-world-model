"""CPU structural and causal tests for prefix attention and policy selection."""
import copy,json
import torch
from common import Predictor
from prefix_policy_predictor import PrefixPredictor,choose_action,policy_rollout,replay_state


def main():
    torch.set_num_threads(2);torch.manual_seed(801)
    base=Predictor(8,4,4,16,2,4).double()
    model=PrefixPredictor(base,8,n_tokens=3,mem_steps=2,mem_grid=2,checkpoint_blocks=False).double()
    z=torch.randn(5,4,8,dtype=torch.double);a=torch.rand(5,3,dtype=torch.double)
    state=model.initial_state(5);before=state.detach().clone();seen=[]
    handle=model.base.blocks[0].sp.register_forward_pre_hook(lambda module,args:seen.append(args[0].shape))
    pred=model(z,a,state);handle.remove()
    assert seen[0][1]==7 # 3 real prefix slots + 4 scene slots in shared self-attention
    assert torch.equal(state,before) and torch.equal(pred[:,0],z)
    assert not any(n.split('.')[0] in ['q','kv','qn','o'] for n,_ in model.named_parameters())
    changed=model(z,a,state+torch.randn_like(state))
    assert not torch.equal(pred,changed),'Prefix must influence predictions immediately'
    residual=torch.randn(5,4,4,8,dtype=torch.double)
    assert torch.equal(model.update_state(state,residual,a),state)
    # Activate residual paths before testing multi-step gradients.
    with torch.no_grad():
        model.update_attention.out_proj.weight.normal_(std=.03)
        model.update_ff[-1].weight.normal_(std=.03)
    model.set_stage('updater');frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    truth=torch.randn(5,4,4,8,dtype=torch.double)
    st=replay_state(model,z,a,[1,3],lambda i:truth[i:i+1],batch_size=2)
    result=model(z[:1],a[:1],st);result.square().mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
    assert all(p.grad is None and torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen)
    # Batch/scalar candidate forwards must agree; a proposed action cannot mutate episode memory.
    zero=torch.zeros(5,dtype=torch.bool);init=[object() for _ in range(5)]
    score=lambda pred,features:pred[:,1:].mean((1,2,3))
    c1,s1,_=choose_action(model,model.initial_state(),z,a,init,score,zero,1)
    c2,s2,_=choose_action(model,model.initial_state(),z,a,init,score,zero,3)
    assert c1==c2 and torch.allclose(s1,s2,atol=1e-12,rtol=1e-12)
    # Labels are revealed after selection: first-action success must halt without an update.
    revealed=[]
    def observe_success(i):revealed.append(i);return True,truth[i:i+1]
    tr,ctx=policy_rollout(model,z,a,init,score,observe_success,budget=5,batch_size=2)
    assert len(tr)==len(revealed)==1 and not ctx
    # With failures, every action is freshly ranked and failed-prefix graph replay is exact.
    tr,ctx=policy_rollout(model,z,a,init,score,lambda i:(False,truth[i:i+1]),budget=4,batch_size=2)
    assert len(tr)==4 and len(ctx)==3 and len({x['local'] for x in tr})==4
    final=replay_state(model,z,a,ctx,lambda i:truth[i:i+1],batch_size=2)
    assert torch.equal(final.detach().cpu(),tr[-1]['state'])
    # Changing an unobserved future must not change first selection.
    altered=truth.clone();altered[c1]+=100
    tr2,_=policy_rollout(model,z,a,init,score,lambda i:(False,altered[i:i+1]),budget=1,batch_size=2)
    assert tr2[0]['local']==tr[0]['local']
    # Gradient checkpointing must preserve outputs and gradients, including frozen-reader input gradients.
    p1=copy.deepcopy(model);p2=copy.deepcopy(model);p2.checkpoint_blocks=True
    p1.zero_grad(set_to_none=True);p2.zero_grad(set_to_none=True)
    for m in [p1,p2]:
        state=replay_state(m,z,a,[1,3],lambda i:truth[i:i+1],batch_size=2)
        m(z[:1],a[:1],state).square().mean().backward()
    maxgrad=max(float((p.grad-dict(p2.named_parameters())[n].grad).abs().max()) for n,p in p1.named_parameters() if p.grad is not None)
    assert maxgrad<1e-10
    print(json.dumps({'passed':True,'prefix_inside_shared_self_attention':True,'active_initial_tokens':True,
        'no_separate_read_adapter':True,'candidate_future_isolation':True,'success_stop_and_no_repeats':True,
        'same_current_policy_residual_replay_exact':True,'full_BPTT_frozen_predictor':True,
        'checkpoint_gradient_max_error':maxgrad,'batch_score_max_error':float((s1-s2).abs().max())},indent=2))

if __name__=='__main__':main()
