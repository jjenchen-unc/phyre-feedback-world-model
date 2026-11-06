"""Current-checkpoint probe diagnosis. No changes to frozen historical sources."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

P = Path(__file__).resolve().parents[3]
R = P / 'results/current_probe_adaptation_20261020'
L = P / 'results/loss_design_iteration_20261019'
sys.path.insert(0, str(L / 'code'))
import loss_runtime as lr

read, write, sha = lr.read, lr.write, lr.sha
VARIANTS = ['q0'] + [f'{loss}_{mode}' for loss in ['l1', 'smooth']
                      for mode in ['raw', 'centered', 'zero']]
ARMS = ['old', 'true_control', 'mixed']


def checked():
    plan = read(R / 'PLAN.json')
    for path, digest in plan['dependencies'].items():
        assert sha(path) == digest, path
    return plan


def make_team():
    import types
    team = lr.make_team('cross')
    original_load = team.load
    def load(self, task, validation=False):
        import numpy as np
        import torch
        folder = R/'pools'/task.replace(':', '_')
        if not (folder/'complete.json').exists():
            assert task in read(R/'PLAN.json')['test']
            return original_load(task, validation)
        receipt = read(folder/'complete.json')
        assert receipt['plan'] == sha(R/'PLAN.json')
        with np.load(folder/'pool.npz') as z:
            self.pool = {k: z[k] for k in z.files}
        self.pool['z0'] = np.load(folder/'z0.npy', mmap_mode='r')
        assert self.pool['z0'].shape == (1024, 196, 128)
        assert len(set(self.pool['action_idx'])) == len({tuple(a) for a in self.pool['action']}) == 1024
        assert (self.pool['status'] != 0).all()
        self.task, self.chunks = task, {}
        for start in range(0, 1024, 32):
            self.chunks[start] = (torch.from_numpy(self.pool['z0'][start:start+32].copy()).cuda().float(),
                                  torch.from_numpy(self.pool['action'][start:start+32]).cuda())
    team.load = types.MethodType(load, team)
    team.truth.parents.insert(0, team.truth.root)
    team.truth.root = R / 'observations/cross'
    team.model.eval().requires_grad_(False)
    team.probe.eval().requires_grad_(False)
    team.head.eval().requires_grad_(False)
    return team


def configure(team, variant):
    kind, mode = ('l1', 'raw') if variant == 'q0' else variant.split('_')
    ck = lr.load(team, 'cross', kind)
    team.input_mode = mode
    team.model.eval().requires_grad_(False)
    return ck


def head(arm):
    import torch
    from common import AttnProbe
    path = Path(read(R / 'PLAN.json')['initial_head']) if arm == 'old' else R / 'heads' / arm / 'probe.pt'
    if arm != 'old':
        done = read(path.parent / 'complete.json')
        assert done['complete'] and done['checkpoint_sha256'] == sha(path)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    h = AttnProbe(ck['D'], ck['S']).cuda()
    h.load_state_dict(ck['state_dict'])
    return h.eval().requires_grad_(False)


def update(team, state, j):
    import torch
    if team.input_mode == 'zero':
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return team.model.update_state(state, torch.zeros(1, 32, 196, 128, device='cuda'),
                                           torch.from_numpy(team.pool['action'][j:j+1]).cuda())
    return team.update(state, j, team.observations([j])[j])


def predict(team, state, indices):
    """Use original 32-candidate forward shapes, including for sampled queries."""
    import torch
    result = {}
    for start in sorted({j // 32 * 32 for j in indices}):
        z, a = team.chunk(start)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = team.model(z, a, state.expand(32, -1, -1)).float()
        for j in indices:
            if j // 32 * 32 == start:
                result[j] = pred[j % 32].detach()
    return torch.stack([result[j] for j in indices])


def state_folder(task, variant, depth):
    return R / 'cache' / task.replace(':', '_') / variant / str(depth)


def save_array(path, value):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.save(f, value)
    tmp.replace(path)


def metrics(scores, labels, tried):
    import numpy as np
    available = np.ones(len(scores), bool)
    available[tried] = False
    y, v = labels[available], scores[available]
    order = np.argsort(-v, kind='stable')
    pos, neg = v[y], v[~y]
    return {
        'auc': float(((pos[:, None] > neg) + .5*(pos[:, None] == neg)).mean()) if len(pos) and len(neg) else None,
        'top1': int(y[order[:1]].any()), 'top8': int(y[order[:8]].any()),
        'n_positive': int(y.sum()), 'n_available': len(y),
    }


def register():
    import numpy as np
    lr.checked()
    src = lr.source()
    source = src.data.checked()
    scope = read(P/'results/probe_cross_training_scope_20261020.json')
    tasks = scope['groups']['original_fit']['scenes']
    assert sha(P/scope['manifest']) == scope['manifest_sha256']
    templates = sorted({t[:5] for t in tasks})
    assert len(templates) == 16
    rng = np.random.default_rng(20261020)
    fit, val = [], []
    for template in templates:
        chosen = rng.permutation(sorted(t for t in tasks if t.startswith(template)))[:6].tolist()
        fit += chosen[:4]
        val += chosen[4:]
    test = source['panels']['cross']['tasks']
    assert len(fit) == 64 and len(val) == 32 and not set(fit) & set(val)
    assert not {t[:5] for t in fit+val} & {t[:5] for t in test}
    initial = P / 'results/probe_adaptation_cross_20261001/heads/predicted_mix_s0/probe.pt'
    dependencies = {str(p): sha(p) for p in sorted((R / 'code').glob('*.py'))}
    dependencies.update({str(p): sha(p) for p in [L/'PLAN.json', L/'code/loss_runtime.py', initial]})
    dependencies[str(P/scope['manifest'])] = scope['manifest_sha256']
    dependencies[str(lr.S/'PLAN.json')] = sha(lr.S/'PLAN.json')
    for kind in ['l1', 'smooth']:
        ck = read(lr.B/'models/cross/uniform/checkpoint_128.json' if kind == 'l1'
                  else L/'models/seed0/cross/smooth/complete.json')
        dependencies[ck['checkpoint']] = ck['checkpoint_sha256']
    plan = dict(fit=fit, validation=val, test=test, templates=templates, variants=VARIANTS,
                initial_head=str(initial), dependencies=dependencies, states=[0, 1, 3, 5],
                query_count=8, future_times=8, steps=2000, batch=128, lr=3e-5,
                weight_decay=.05, clip=1., seed=20261020,
                sampling='Scene selection independent of outcomes/cache. Uniform untried query permutation; no Near or success filter. Online current-checkpoint/old-head support selection, stop on success.',
                representation='FP32 arrays retain exact BF16 predictor outputs and observed FP16 values; original 32-candidate forward batches.',
                training='Common head, uniform scene then available variant then state then query/time/dynamic object. Identical target draws. True control: observed only; mixed: exactly half observed, half predicted.',
                loss='10*xy L1/256 + doubled-angle L1 on bars. Geometry targets only; fixed final step.',
                heldout_scope='Validation scenes disjoint for this adaptation; upstream predictors/probe may have seen them. Test templates disjoint.',
                scene_universe='Original cross probe 400 training scenes, 25/template x16; 4 fit +2 validation per template, deterministic outcome-independent draw. All verified official cross fold1 train.',
                gate='On 32 validation scenes: mixed predicted geometry loss strictly lower and full-pool macro AUC strictly higher than BOTH old and true_control; top8 coverage no lower than either. Equal scene weights, then equal available variant/state weights. AUC only jointly defined states. Exploratory gate, not significance.',
                closed_loop='If gate passes: cross100, all7 variants, true_control and mixed; own first and old-Q0 common first, 9 attempts. Reuse original-head reference. No updater training.',
                limits=['One fold and seed', 'Old-head support distribution during fit; new-head policy distribution tested only after gate', 'No success-loss training', 'Available later states conditional on prior failures'])
    assert not (R/'PLAN.json').exists()
    write(R/'PLAN.json', plan)
    print('REGISTERED', len(fit), len(val), flush=True)
