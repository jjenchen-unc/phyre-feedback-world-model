"""Validation-only strict kernels, except unchanged disjoint pooling backward.

No predictor/scorer forward is replaced. Torch rejects adaptive-pool backward
under strict determinism even for disjoint bins. Relax its guard only for that
operator after checking that input dimensions divide output bins exactly.
This is an explicitly bounded diagnostic exception, not a universal determinism
claim and not enabled in registered training jobs.
"""
from contextlib import contextmanager
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class DisjointPoolBackward(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.shapes = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._adaptive_avg_pool2d_backward.default:
            grad, original = args[:2]
            ih, iw = original.shape[-2:]
            oh, ow = grad.shape[-2:]
            if ih % oh or iw % ow:
                raise RuntimeError('Only disjoint adaptive-pool bins are allowed in this diagnostic')
            self.calls += 1
            self.shapes.add((ih, iw, oh, ow))
            enabled = torch.are_deterministic_algorithms_enabled()
            warn = torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                torch.use_deterministic_algorithms(False)
                return func(*args, **kwargs)
            finally:
                torch.use_deterministic_algorithms(enabled, warn_only=warn)
        return func(*args, **kwargs)


@contextmanager
def strict_except_disjoint_pool():
    before = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    mode = DisjointPoolBackward()
    try:
        torch.use_deterministic_algorithms(True)
        with mode:
            yield mode
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn)
