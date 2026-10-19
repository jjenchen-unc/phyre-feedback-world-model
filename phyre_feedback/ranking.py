"""Inference-only ranking with fewer CPU/GPU transfers and reference arithmetic.

The exact backend keeps the predictor batch size (32), per-object probe shapes,
FP32 probe, NumPy geometry and float64 log score unchanged. Only initial queries
are cached. Predictions, memory residuals and model weights are never cached.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .metrics import log_score
from .geometry import surface_dist


class PreparedProbeReadout:
    """Reuse initial object queries and transfer all readouts together.

    Unlike merging attention queries into a larger batch, preserving each
    probe.read shape avoids the rounding changes seen in the earlier prototype.
    The layout belongs to one fixed candidate chunk, not to a predictor state.
    """

    def __init__(self, features, steps, device):
        if not features or steps <= 0:
            raise ValueError('Nonempty features and positive time dimension required')
        self.initial = [np.array(f[:1], copy=True) for f in features]
        self.steps = steps
        # Canonicalize an unindexed 'cuda' to its allocated local CUDA device.
        self.device = torch.empty(0, device=device).device
        self.objects, queries = [], []
        for j, f in enumerate(self.initial):
            if f.ndim != 3 or f.shape[0] != 1 or f.shape[-1] != 14 or f.dtype != np.float32:
                raise ValueError('Expected float32 features [time, objects, 14]')
            for obj in range(f.shape[1]):
                if f[0, obj, 13] >= .5 or f[0, obj, 11] >= .5:
                    continue
                self.objects.append((j, obj))
                queries.append(np.repeat(f[:1, obj], steps, axis=0))
        self.queries = (torch.from_numpy(np.stack(queries)).to(self.device)
                        if queries else None)

    @torch.no_grad()
    def __call__(self, probe, prediction, rule):
        if probe.training:
            raise ValueError('The ranking probe must be in eval mode')
        b, nt, ns, _ = prediction.shape
        if b != len(self.initial) or nt != self.steps or prediction.device != self.device:
            raise ValueError('Prediction shape/device differs from prepared candidates')
        # Fixed mode is important even if a caller has predictor autocast enabled.
        with torch.autocast(prediction.device.type, enabled=False):
            kv = probe.inp(prediction.float().reshape(b * nt, ns, -1)) + probe.space
            outputs = []
            for i, (candidate, _) in enumerate(self.objects):
                xy, angle = probe.read(kv[candidate*nt:(candidate+1)*nt], self.queries[i])
                outputs.append(torch.cat((xy, angle), dim=-1))
            # One D2H copy per candidate chunk. Object forwards retain the exact
            # reference batch/sequence shapes; concatenation performs no arithmetic.
            readouts = torch.stack(outputs).cpu().numpy() if outputs else None
        frames = [np.repeat(f, nt, axis=0) for f in self.initial]
        for i, (candidate, obj) in enumerate(self.objects):
            # Match the legacy contiguous [time,2] arrays before NumPy arithmetic.
            xy = readouts[i, :, :2].copy()
            angle = readouts[i, :, 2:].copy()
            frames[candidate][:, obj, :2] = xy / 256
            frames[candidate][:, obj, 2] = (
                np.arctan2(angle[:, 1], angle[:, 0]) / 2) / (2 * np.pi) % 1
        distances = np.stack([surface_dist(f) for f in frames])
        return log_score(distances, rule)


class PreparedScorer:
    """Tensor scorer interface for an existing policy loop, with a bounded cache.

    The key contains the actual initial feature values, not Python object IDs;
    reusing an instance across scenes cannot accidentally reuse another scene's
    geometry. Only static query inputs are cached, never scores or predictions.
    """

    def __init__(self, probe, rule, max_layouts=64):
        if max_layouts < 1:
            raise ValueError('max_layouts must be positive')
        self.probe, self.rule = probe, tuple(rule)
        self.max_layouts = max_layouts
        self.layouts = OrderedDict()

    @torch.no_grad()
    def __call__(self, prediction, features):
        fingerprint = hashlib.sha256()
        for f in features:
            initial = np.ascontiguousarray(f[:1])
            fingerprint.update(str((initial.shape, initial.dtype.str)).encode())
            fingerprint.update(initial.tobytes())
        key = (prediction.shape[1], str(prediction.device), fingerprint.digest())
        if key not in self.layouts:
            self.layouts[key] = PreparedProbeReadout(features, prediction.shape[1], prediction.device)
            if len(self.layouts) > self.max_layouts:
                self.layouts.popitem(last=False)
        self.layouts.move_to_end(key)
        values = self.layouts[key](self.probe, prediction, self.rule)
        return torch.from_numpy(values).to(prediction.device)


@dataclass
class Ranking:
    scores: np.ndarray
    selected: int | None
    prediction: torch.Tensor | None


class CandidateRanker:
    """Reusable base/Q0/current-memory ranking for one scene's fixed pool.

    Each call performs fresh predictor forwards. For memory, pass the CURRENT
    state; for a base/Q0/TTA predictor omit state. All tried actions are masked,
    and equal scores are resolved by original candidate order.
    """

    def __init__(self, model, probe, rule, z0, actions, features, batch_size=32):
        if batch_size != 32:
            raise ValueError('Exact inference keeps the validated predictor batch size 32')
        if len(z0) == 0 or len(z0) != len(actions) or len(z0) != len(features):
            raise ValueError('Nonempty, aligned candidate inputs required')
        if z0.device != actions.device:
            raise ValueError('Latents and actions must use the same device')
        if model.training or probe.training:
            raise ValueError('CandidateRanker is inference-only; call model.eval()/probe.eval()')
        self.model, self.probe, self.rule = model, probe, tuple(rule)
        self.z0, self.actions = z0, actions
        self.features = [np.array(f[:1], copy=True) for f in features]
        self.batch_size = batch_size
        self.layouts = {}

    @torch.no_grad()
    def rank(self, state=None, tried=None):
        if self.model.training or self.probe.training:
            raise ValueError('Inference models must remain in eval mode')
        if tried is None:
            tried = np.zeros(len(self.actions), dtype=bool)
        elif isinstance(tried, torch.Tensor):
            tried = tried.detach().cpu().numpy()
        tried = np.asarray(tried)
        if tried.shape != (len(self.actions),) or tried.dtype != bool:
            raise ValueError('tried must be a boolean mask of the candidate pool')
        if state is not None and (state.ndim != 3 or state.shape[0] != 1):
            raise ValueError('Memory state must have shape [1, tokens, width]')
        chunks = []
        best, selected, chosen_prediction = -np.inf, None, None
        for start in range(0, len(self.actions), self.batch_size):
            end = min(start+self.batch_size, len(self.actions))
            # CUDA matches the original BF16 predictor. CPU is for structural tests.
            with torch.autocast(self.z0.device.type, dtype=torch.bfloat16,
                                enabled=self.z0.device.type == 'cuda'):
                if state is None:
                    pred = self.model(self.z0[start:end], self.actions[start:end])
                else:
                    pred = self.model(self.z0[start:end], self.actions[start:end],
                                      state.expand(end-start, -1, -1))
            if start not in self.layouts:
                self.layouts[start] = PreparedProbeReadout(
                    self.features[start:end], pred.shape[1], pred.device)
            scores = self.layouts[start](self.probe, pred, self.rule)
            if scores.shape != (end-start,) or not np.isfinite(scores).all():
                raise FloatingPointError('Candidate scores must be finite and aligned')
            chunks.append(scores)
            available = scores.copy()
            available[tried[start:end]] = -np.inf
            local = int(np.argmax(available))
            value = float(available[local])
            if value > best:
                best, selected = value, start+local
                chosen_prediction = pred[local:local+1].clone()
        return Ranking(np.concatenate(chunks), selected, chosen_prediction)


def static_scores(model, probe, rule, z0, actions, features):
    """One-shot convenience wrapper; retain CandidateRanker for repeated trials."""
    return CandidateRanker(model, probe, rule, z0, actions, features).rank().scores
