"""Validation-only numerical controls; never changes model/scorer forwards.

Strict Torch algorithms are retained. Pool backward can either execute the
original CUDA operator under an explicit guard exception, or the CPU operator.
The latter changes accumulation order but not the pooling derivative, and is
applied to BOTH arms to isolate implementation effects from CUDA nondeterminism.
"""
from contextlib import contextmanager
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class PoolBackwardControl(TorchDispatchMode):
    def __init__(self, backend):
        super().__init__()
        if backend not in ['cpu','cuda']: raise ValueError(backend)
        self.backend=backend;self.calls=0;self.shapes=set()

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        if func==torch.ops.aten._adaptive_avg_pool2d_backward.default:
            grad,original=args[:2]
            self.calls+=1
            self.shapes.add((*original.shape[-2:],*grad.shape[-2:]))
            if self.backend=='cpu':
                # ATen pool backward uses the original input's shape, not values.
                shape_only=torch.empty(original.shape,dtype=original.dtype,device='cpu')
                return func(grad.cpu(),shape_only).to(original.device)
            enabled=torch.are_deterministic_algorithms_enabled()
            warn=torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                torch.use_deterministic_algorithms(False)
                return func(*args,**kwargs)
            finally:
                torch.use_deterministic_algorithms(enabled,warn_only=warn)
        return func(*args,**kwargs)


@contextmanager
def controlled_backward(pool_backend='cpu',strict=True):
    before=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    mode=PoolBackwardControl(pool_backend)
    try:
        torch.use_deterministic_algorithms(strict)
        with mode:yield mode
    finally:
        torch.use_deterministic_algorithms(before,warn_only=warn)
