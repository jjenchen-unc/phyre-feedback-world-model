"""Verify the diagnostic changes only pooling backward and restores global flags."""
import torch
from torch.nn import functional as F
from updater_validation_numerics import controlled_backward


def main():
    torch.set_num_threads(2);torch.manual_seed(701)
    results=[]
    for grid in [14,16,18,20]:
        values=torch.randn(2,3,grid,grid)
        cotangent=torch.randn(2,3,10,10)
        cpu=values.clone().requires_grad_(True)
        F.adaptive_avg_pool2d(cpu,10).backward(cotangent)
        cuda=values.cuda().requires_grad_(True)
        original=F.adaptive_avg_pool2d(cuda,10)
        original.backward(cotangent.cuda());original_grad=cuda.grad.clone()
        obtained=[]
        for _ in range(2):
            cuda.grad=None
            with controlled_backward('cpu') as control:
                output=F.adaptive_avg_pool2d(cuda,10)
                assert torch.equal(output,original)
                output.backward(cotangent.cuda())
                assert control.calls==1
                assert torch.are_deterministic_algorithms_enabled()
            assert not torch.are_deterministic_algorithms_enabled()
            assert torch.equal(cuda.grad.cpu(),cpu.grad)
            torch.testing.assert_close(cuda.grad,original_grad)
            obtained.append(cuda.grad.clone())
        assert torch.equal(*obtained)
        results.append({'grid':grid,'forward_exact':True,'cpu_gradient_exact':True,
                        'cuda_gradient_max_abs':float((obtained[0]-original_grad).abs().max())})
    try:
        with controlled_backward('cpu'):
            raise RuntimeError('test restoration')
    except RuntimeError:pass
    assert not torch.are_deterministic_algorithms_enabled()
    print('PASS',results,flush=True)


if __name__=='__main__':main()
