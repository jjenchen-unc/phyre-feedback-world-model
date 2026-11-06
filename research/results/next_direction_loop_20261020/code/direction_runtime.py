"""Bounded three-round analysis; historical sources and caches stay immutable."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

P = Path(__file__).resolve().parents[3]
R = P/'results/next_direction_loop_20261020'
A = P/'results/current_probe_adaptation_20261020'
sys.path.insert(0, str(A/'code'))
import adapt_runtime as ar
read, write, sha = ar.read, ar.write, ar.sha
ARMS = ['q0_mixed', 'memory_mixed', 'memory_predicted', 'memory_task_objects']
SEEDS = [20261021, 20261022]


def checked():
    plan = read(R/'PLAN.json')
    for path, digest in plan['dependencies'].items():
        assert sha(path) == digest, path
    return plan


def folder(task, variant, depth):
    return A/'cache'/task.replace(':', '_')/variant/str(depth)


def load_head(name, smoke=False):
    import torch
    from common import AttnProbe
    if name in ['old', 'previous_mixed']:
        return ar.head('old' if name == 'old' else 'mixed')
    d = R/('smoke_heads' if smoke else 'heads')/name
    rec = read(d/'complete.json')
    assert rec['complete'] and rec['plan'] == sha(R/'PLAN.json')
    assert sha(d/'probe.pt') == rec['checkpoint_sha256']
    ck = torch.load(d/'probe.pt', map_location='cpu', weights_only=False)
    h = AttnProbe(ck['D'], ck['S']).cuda()
    h.load_state_dict(ck['state_dict'])
    return h.eval().requires_grad_(False)


def rank_metrics(scores, labels, tried):
    mask = np.ones(len(labels), bool)
    mask[tried] = False
    v, y = np.asarray(scores)[mask], np.asarray(labels)[mask]
    assert np.isfinite(v).all()
    order = np.argsort(-v, kind='stable')
    pos, neg = v[y], np.sort(v[~y])
    auc = float((np.searchsorted(neg, pos, side='left')+np.searchsorted(neg, pos, side='right')).sum()/(2*len(pos)*len(neg))) if len(pos) and len(neg) else None
    return {**{f'top{k}': int(y[order[:k]].any()) for k in [1, 3, 5, 8]}, 'auc': auc,
            'positive': int(y.sum()), 'available': len(y)}


def mean_metrics(rows):
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
            if any(r[k] is not None for r in rows) else None for k in ['top1', 'top3', 'top5', 'top8', 'auc']}


def register():
    prior = ar.checked()
    assert read(A/'FINAL_SUMMARY.json')['complete']
    discovery, confirmation = [], []
    for template in prior['templates']:
        tasks = sorted(t for t in prior['validation'] if t.startswith(template))
        assert len(tasks) == 2
        discovery.append(tasks[0]); confirmation.append(tasks[1])
    deps = {str(p): sha(p) for p in sorted((R/'code').glob('*.py'))}
    for p in [A/'PLAN.json', A/'DATA_AUDIT.json', A/'HEAD_AUDIT.json', A/'code/adapt_runtime.py', A/'code/evaluate.py']:
        deps[str(p)] = sha(p)
    deps.update(prior['dependencies'])
    plan = dict(fit=prior['fit'], discovery=discovery, confirmation=confirmation,
                dependencies=deps, arms=ARMS, seeds=SEEDS, steps=2000, batch=128,
                lr=3e-5, weight_decay=.05, clip=1., initial_head=prior['initial_head'],
                round1='Cached full1024 ranks: Q0 versus memory versus same-loss zero on IDENTICAL first-failure history. Select raw/centered L1/Smooth using FIT scenes only: top8 gain vszero, then vsQ0, then top1 vszero, then AUC vszero. Deeper oldpolicy states reported separately, not causal history controls.',
                round2='8 small ranking-head fits: Q0-domain50/50, selected-memory-domain50/50, selected-memory predicted-only, selected-memory50/50 goal-object-only. Two target-sampling seeds, identical head initialization. No new predictor/updater fits.',
                task_object_definition='Initial dynamic GREEN or BLUE; static PURPLE remains fixed geometry. Only sampled object changes from generic domain-mixed control; same scene/state/query/time draw.',
                ranking='Same1024pool, oldpolicy cached states at depth1 primary. Q0-after-firstfailure baseline masks sameattempt; zero shares sameactualfirstaction. Static top1/3/5/8, notclosedloop success.',
                round3='Pick memory arm on discovery by worst-seed top8 delta then mean top8 then mean top1 then AUC. Promote only top8 nonnegative for bothseeds, meanpositive, top1 nonnegative for bothseeds. Confirm selected arm AND memory_mixed comparator on remaining16scenes; Q0-domain tested separately. No branch can train on discovery/confirmation.',
                interpretation='AUC/geometry auxiliary, no joint geometry/AUC gate. Check memory-vs-zero and memory-vs-Q0 on same failed histories before attributing benefit to residual content.',
                bounds='At most8newheads,2000steps each,2seeds. No new simulation/encoding/100kselection orfullclosedloop in this screening. Reuse exact capturedtruth/predictions; no latentcache approximation.',
                limitations=['Onefold; two probe samplingseeds, not independentworldmodels', '16+16scenes frompreviouslyexamined adaptationvalidation; not untouchedtest', 'First-failure memorymetrics conditional on Q0 firstfailure; reportdenominators', 'Fixedhistory intervention does notmeasureclosedloopdeployment gain'])
    assert not (R/'PLAN.json').exists()
    write(R/'PLAN.json', plan)
    print('REGISTERED_DIRECTION_LOOP', flush=True)
