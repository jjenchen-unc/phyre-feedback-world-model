"""Batched frozen-probe readout; keep legacy geometry and score arithmetic.

Use only with eval-mode probes. Object queries never attend to each other: each
cross-attends to the same latent keys/values, followed by pointwise layers.
The predictor candidate batch and current-token residual definition are unchanged.
"""
from contextlib import contextmanager

import numpy as np
import torch

from common import surface_dist
from attribution_a_metrics import log_score


def read_objects(probe, kv, queries):
    """kv [N,S,D], queries [N,O,14]; independent queries share latent KV."""
    if probe.training:
        raise ValueError("Batched readout requires an eval-mode probe")
    h = probe.qin(queries)
    for att, mlp, ln in zip(probe.att, probe.mlp, probe.ln):
        # Keep need_weights=True, matching the reference attention arithmetic.
        h = h + att(ln(h), kv, kv)[0]
        h = h + mlp(h)
    out = probe.out(h)
    return torch.sigmoid(out[..., :2]) * 256, torch.tanh(out[..., 2:])


@torch.no_grad()
def readout(probe, pred, features, rule):
    b, nt, ns, _ = pred.shape
    if len(features) != b:
        raise ValueError("One initial feature record is required per candidate")
    dynamic = [np.flatnonzero((f[0, :, 13] < .5) & (f[0, :, 11] < .5))
               for f in features]
    count = max(map(len, dynamic), default=0)
    with torch.autocast(pred.device.type, enabled=False):
        if count:
            initial = np.zeros((b, count, 14), dtype=np.float32)
            for j, ids in enumerate(dynamic):
                initial[j, :len(ids)] = features[j][0, ids]
            q = torch.from_numpy(initial).to(pred.device)
            q = q[:, None].expand(-1, nt, -1, -1).reshape(b * nt, count, 14)
            kv = probe.inp(pred.float().reshape(b * nt, ns, -1)) + probe.space
            xy, angle = read_objects(probe, kv, q)
            # Two transfers per candidate batch, instead of two per object/action.
            xy = xy.reshape(b, nt, count, 2).cpu().numpy()
            angle = angle.reshape(b, nt, count, 2).cpu().numpy()
    distances, positions = [], []
    for j, f in enumerate(features):
        pf = np.repeat(f[:1], nt, 0).copy()
        for k, obj in enumerate(dynamic[j]):
            pf[:, obj, :2] = xy[j, :, k] / 256
            a = angle[j, :, k]
            pf[:, obj, 2] = (np.arctan2(a[:, 1], a[:, 0]) / 2) / (2 * np.pi) % 1
        distances.append(surface_dist(pf))
        positions.append(pf[:, :, :2] * 256)
    ds = np.stack(distances)
    return {'distance': ds, 'xy': np.stack(positions), 'log_score': log_score(ds, rule)}


@contextmanager
def accelerated_scoring(prefix_module):
    """Opt-in for a NEW run; never alter frozen source files or run manifests.

    Applies to both stage-two support selection and evaluation via score_function.
    Geometry supervision, policy replay and gradients are untouched.
    """
    previous = prefix_module.f.readout
    prefix_module.f.readout = readout
    try:
        yield
    finally:
        prefix_module.f.readout = previous
