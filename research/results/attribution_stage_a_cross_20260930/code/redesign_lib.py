"""Shared utilities for the registered R0--R4 memory experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from common import Predictor, MemoryPredictor, AttnProbe

ROOT = Path('/mnt/meg/jjenchen/phyre-unc')
PROJECT = Path('/path/to/phyre-unc')
BASE = ROOT / 'runs/cross_f1_base_s16r16_pca128/predictor.pt'
PROBE = PROJECT / 'results/probes_pca128_cross/ball_cross_template_f1_full.pt'
OLD_MEMORY = ROOT / 'runs/mem_cross_f1_residual_hires_clusters_gate/memory.pt'
PACKS = ['cross_f1_train_clusters_pca128', 'cross_f1_train_s16r16_pca128']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); tmp.replace(path)


class Store:
    """Deduplicated, lazy union; never duplicates the large latent files."""
    def __init__(self, include_features=True):
        self.lat = [np.load(ROOT / 'packs' / (p + '_latents.npy'), mmap_mode='r') for p in PACKS]
        self.meta = []
        for name in PACKS:
            with np.load(ROOT / 'packs' / (name + '_meta.npz')) as packed:
                self.meta.append({key: packed[key] for key in packed.files
                                  if include_features or key not in ['features', 'mask']})
        self.refs, seen = [], set()
        for p, m in enumerate(self.meta):
            for i, file in enumerate(m['files']):
                if str(file) not in seen:
                    self.refs.append((p, i)); seen.add(str(file))
        self.tasks = np.array([self.meta[p]['tasks'][i] for p, i in self.refs])
        self.files = np.array([self.meta[p]['files'][i] for p, i in self.refs])
        self.labels = np.array([self.meta[p]['solved'][i] for p, i in self.refs], bool)
        self.actions = np.stack([self.meta[p]['action'][i] for p, i in self.refs])
        self.by = {t: np.flatnonzero(self.tasks == t) for t in sorted(set(self.tasks))}

    def z(self, ids):
        return np.stack([self.lat[self.refs[int(i)][0]][self.refs[int(i)][1]] for i in ids])

    def feature(self, i):
        p, j = self.refs[int(i)]
        return self.meta[p]['features'][j][:, self.meta[p]['mask'][j]]


def load_models(device='cuda', memory=None):
    pc = torch.load(BASE, map_location='cpu', weights_only=False); a = pc['args']
    base = Predictor(pc['Din'], pc['T'], pc['S'], a['d'], a['layers'], a['heads']).to(device)
    base.load_state_dict(pc['state_dict']); base.eval()
    qc = torch.load(PROBE, map_location='cpu', weights_only=False)
    probe = AttnProbe(qc['D'], qc['S']).to(device)
    probe.load_state_dict(qc['state_dict']); probe.eval()
    for p in probe.parameters(): p.requires_grad_(False)
    model = None
    if memory:
        mc = torch.load(memory, map_location='cpu', weights_only=False)
        assert mc['base'] == str(BASE) and not mc.get('unfreeze', False)
        model = MemoryPredictor(base, pc['Din'], mode=mc['mode'], mem_steps=mc['mem_steps'],
                                mem_grid=mc['mem_grid'], per_attempt=mc['per_attempt'], gate_empty=True).to(device)
        missing, unexpected = model.load_state_dict(mc['state_dict'], strict=False)
        assert not unexpected and all(k.startswith('base.') for k in missing)
        model.eval()
    return base, probe, qc['rule'], pc, model


def torch_surface(f):
    """Equivalent to common.surface_dist for one [time, object, 14] tensor."""
    g = int(torch.nonzero(f[0, :, 9] > .5)[0, 0])
    goals = torch.nonzero((f[0, :, 10] > .5) | (f[0, :, 11] > .5)).flatten().tolist()
    cg, rg = f[:, g, :2] * 256, f[0, g, 3] * 128
    distances = []
    for o in goals:
        co, length = f[:, o, :2] * 256, f[0, o, 3] * 256
        if int(f[0, o, 4:8].argmax()) == 1:
            theta = f[:, o, 2] * (2 * torch.pi)
            u = torch.stack([theta.cos(), theta.sin()], -1)
            t = ((cg - co) * u).sum(-1).clamp(-length / 2, length / 2)
            distances.append(torch.linalg.vector_norm(cg - co - t[:, None] * u, dim=-1) - rg)
        else:
            distances.append(torch.linalg.vector_norm(cg - co, dim=-1) - rg - length / 2)
    return torch.stack(distances).amin(0)


def readout(pr, features, probe, rule, details=False):
    """Frozen differentiable soft scorer. Query inputs use initial geometry only."""
    assert rule[0] == 'soft'
    b, nt, ns, _ = pr.shape
    kv = probe.inp(pr.float().reshape(b * nt, ns, -1)) + probe.space
    scores, positions, distances = [], [], []
    for j, f in enumerate(features):
        f = torch.as_tensor(f, device=pr.device)
        objects = []
        for o in range(f.shape[1]):
            initial = f[0, o].expand(nt, -1)
            if f[0, o, 13] < .5 and f[0, o, 11] < .5:
                xy, angle = probe.read(kv[j * nt:(j + 1) * nt], initial)
                theta = torch.atan2(angle[:, 1], angle[:, 0]) / (4 * torch.pi) % 1
                objects.append(torch.cat([xy / 256, theta[:, None], initial[:, 3:]], -1))
            else:
                objects.append(initial)
        pf = torch.stack(objects, 1)
        d = torch_surface(pf)
        scores.append((.5 * (1 + torch.tanh((rule[1] - d) / (2 * rule[2])))).sum())
        if details:
            g = int(torch.nonzero(f[0, :, 9] > .5)[0, 0])
            positions.append(pf[:, g, :2] * 256); distances.append(d)
    scores = torch.stack(scores)
    return (scores, torch.stack(positions), torch.stack(distances)) if details else scores


def context_memory(model, store, ids, device='cuda'):
    if ids:
        z = torch.from_numpy(store.z(ids)).to(device).float()[None]
        a = torch.from_numpy(store.actions[ids]).to(device)[None]
        y = torch.from_numpy(store.labels[ids].astype(np.int64)).to(device)[None]
        mask = torch.ones(1, len(ids), dtype=torch.bool, device=device)
    else:
        z = torch.zeros((1, 1) + store.lat[0].shape[1:], device=device)
        a = torch.zeros(1, 1, 3, device=device); y = torch.zeros(1, 1, device=device)
        mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
        residual = model.context_residual(z, a).float()
        return model.encode_memory(z, a, y, mask, residual)


@torch.no_grad()
def score_ids(base, probe, rule, store, ids, model=None, context=(), batch=16):
    memory = context_memory(model, store, list(context)) if model is not None else None
    out = []
    for start in range(0, len(ids), batch):
        ii = ids[start:start + batch]
        z = torch.from_numpy(store.z(ii)[:, 0].copy()).cuda().float()
        a = torch.from_numpy(store.actions[ii]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pr = (base(z, a) if model is None else model(z, a, memory[0].expand(len(ii), -1, -1), memory[1].expand(len(ii), -1))).float()
        out.extend(readout(pr, [store.feature(i) for i in ii], probe, rule).cpu().tolist())
    return np.asarray(out)
