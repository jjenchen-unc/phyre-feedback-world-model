"""Paired speed/equivalence checks against completed, immutable PHYRE runs."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
MEM = PROJECT / 'results/small_official_20261010'
sys.path.insert(0, str(MEM / 'code'))
sys.path.insert(0, str(MEM / 'recovery_20261010'))
import recovery_runner as m
from prefix_policy_predictor import choose_action
from batched_readout import accelerated_scoring, readout

p = m.p
OUT = PROJECT / 'results/performance_20261012'


def save(name, data):
    m.write(OUT / (name + '.json'), data)


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


@torch.no_grad()
def scores(model, z0, actions, features, score, batch):
    result = []
    for start in range(0, len(actions), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(z0[start:start + batch], actions[start:start + batch])
        result.append(score(pred, features[start:start + batch]).cpu().numpy())
    return np.concatenate(result)


def comparison(a, b):
    order_a, order_b = np.argsort(-a, kind='stable'), np.argsort(-b, kind='stable')
    return {'max_abs': float(np.max(np.abs(a - b))),
            'mean_abs': float(np.mean(np.abs(a - b))),
            'argmax_equal': bool(order_a[0] == order_b[0]),
            'top20_equal': bool(np.array_equal(order_a[:20], order_b[:20])),
            'ranking_equal': bool(np.array_equal(order_a, order_b))}


def inference(panel):
    # One deterministic task per template, preferring a long failed prefix to
    # exercise current-state residuals. No task is selected by measured speed.
    folder = MEM / 'evaluation_fixed' / panel / 'residual_s0'
    tasks = m.read(MEM / 'selection.json')['panels'][panel]['selected']
    templates = sorted({t[:5] for t in tasks})
    if panel == 'within':
        templates = templates[::5]
    selected = []
    for template in templates:
        ts = [t for t in tasks if t.startswith(template)]
        selected.append(min(ts, key=lambda t: (
            -m.read(folder / t.replace(':', '_') / 'result.json')['attempts'], t)))
    ref, model, head, rule, _ = m.model_for(panel, 'residual', 0)
    model.eval().requires_grad_(False)
    score = p.score_function(head, rule)
    static, traces = [], []
    old_readout = p.f.readout
    for task in selected:
        d, status, path = m.input_case(panel, task)
        z0 = torch.from_numpy(d['z0']).cuda().float()
        actions = torch.from_numpy(d['action']).cuda()
        features = [v[None] for v in d['features']]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(z0[:32], actions[:32])
        # Warm both scorer paths, then interleave measurements on identical pred.
        old_readout(head, pred, features[:32], rule)
        readout(head, pred, features[:32], rule)
        component = {'legacy': [], 'batched': []}
        for repeat in range(3):
            for name, fn in [('legacy', old_readout), ('batched', readout)]:
                _, sec = timed(lambda: fn(head, pred, features[:32], rule))
                component[name].append(sec)
        with np.load(MEM / 'evaluation_fixed' / panel / 'q0_s0' /
                     task.replace(':', '_') / 'scores.npz') as z:
            expected = z['q0'].copy()
        baseline, sec = timed(lambda: scores(model, z0, actions, features, score, 32))
        np.testing.assert_array_equal(baseline, expected)
        entry = {'task': task, 'component_seconds': component, 'legacy_seconds': sec,
                 'batches': {}}
        for batch in [32, 64, 128]:
            torch.cuda.reset_peak_memory_stats()
            with accelerated_scoring(p):
                output, elapsed = timed(lambda: scores(model, z0, actions, features, score, batch))
            entry['batches'][str(batch)] = {
                'seconds': elapsed, 'speedup': sec / elapsed,
                'peak_GiB': torch.cuda.max_memory_allocated() / 2**30,
                **comparison(expected, output)}
        static.append(entry)
        # Test the actual autoregressive policy, without forcing reference actions.
        # B32 preserves the predictor's kernel shapes and selected predictions.
        saved = folder / task.replace(':', '_')
        with np.load(saved / 'scores.npz') as z:
            expected_actions, reference_scores = z['selected'].copy(), z['scores'].copy()
        state = model.initial_state().clone()
        tried = torch.zeros(1024, dtype=torch.bool, device='cuda')
        actual, drift = [], []
        start = time.perf_counter()
        with torch.no_grad(), accelerated_scoring(p):
            for attempt in range(20):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    chosen, sc, prediction = choose_action(model, state, z0, actions,
                                                          features, score, tried)
                actual.append(chosen)
                assert chosen == int(expected_actions[attempt]), (task, attempt, chosen, expected_actions)
                drift.append(comparison(reference_scores[attempt], sc.cpu().numpy()))
                tried[chosen] = True
                if status[chosen] > 0 or attempt == 19:
                    break
                obs = MEM / 'assets_fixed' / panel / 'observations' / (
                    task.replace(':', '_') + f"_{int(d['action_idx'][chosen]):06d}.npz")
                # All reference observations already exist; never simulate a new
                # action or modify completed-run assets in this benchmark.
                with np.load(obs) as z:
                    truth = torch.from_numpy(z['latents'][None].copy()).cuda().float()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    state = model.update_state(state, truth - prediction, actions[chosen:chosen + 1])
        assert actual == expected_actions.tolist()
        torch.cuda.synchronize()
        traces.append({'task': task, 'attempts': len(actual), 'selected_equal': True,
                       'seconds': time.perf_counter() - start, 'score_comparisons': drift})
        save('inference_' + panel + '_progress', {'static': static, 'traces': traces})
        print('INFERENCE', panel, task, entry, 'trace', len(actual), flush=True)
    result = {'complete': True, 'panel': panel, 'static': static, 'traces': traces,
              'code': hashes(), 'device': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'all_selected_actions_equal': True}
    save('inference_' + panel, result)


def flat_grad(model):
    return torch.cat([v.grad.detach().float().reshape(-1).cpu() for v in model.parameters()
                      if v.requires_grad and v.grad is not None])


def train_pass(model, losses, checkpoint, fast, stage):
    model.checkpoint_blocks = checkpoint
    opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                            lr=3e-4, weight_decay=.05)
    elapsed, events, gradients = [], [], []
    torch.cuda.reset_peak_memory_stats()
    context = accelerated_scoring(p)
    if fast:
        context.__enter__()
    try:
        for loss_fn in losses:
            def step():
                opt.zero_grad(set_to_none=True)
                loss, event = loss_fn()
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                return event
            event, sec = timed(step)
            elapsed.append(sec)
            events.append(event)
            gradients.append(flat_grad(model))
            print('TRAIN_STEP', stage, checkpoint, fast, sec, event, flush=True)
    finally:
        if fast:
            context.__exit__(None, None, None)
    return {'seconds': elapsed, 'events': events,
            'peak_GiB': torch.cuda.max_memory_allocated() / 2**30}, gradients


def training(panel):
    if panel == 'within':
        m.configure_within()
        root = m.W
    else:
        root = m.OLD
    manifest = m.read(root / 'manifest.json')
    results = {}
    for stage in ['q0', 'residual']:
        ref, model, head, rule, _ = m.model_for(panel, stage, 0)
        model.train()
        probe_ck = torch.load(m.PROBES[panel], map_location='cpu', weights_only=False)
        probe = m.AttnProbe(probe_ck['D'], probe_ck['S']).cuda()
        probe.load_state_dict(probe_ck['state_dict'])
        probe.eval().requires_grad_(False)
        init = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stage == 'q0':
            source = p.f.Pack(manifest['stage1_pack'])
            ids_list = p.old.epoch_indices(len(source.actions), 0, 0)[:8].tolist()
            def qloss(ids):
                loss, l1, geom = p.old.q0_loss(model, probe, source, ids)
                return loss, {'ids': ids, 'loss': float(loss.detach()),
                              'l1': float(l1.detach()), 'geometry': float(geom.detach())}
            losses = [lambda ids=ids: qloss(ids) for ids in ids_list]
        else:
            source = p.f.Store()
            rows = [r for r in manifest['rows'] if r['split'] == 'train']
            # Fixed chronological training draws, two equally weighted episodes
            # per step, including zero-support episodes and full replay gradients.
            def uloss(slot):
                # Backwards remain sequential to match actual training memory use.
                raise RuntimeError('use training_slots for updater')
            results[stage] = training_slots(model, probe, head, rule, source, rows, init)
            del source, model, ref, head, probe
            gc.collect(); torch.cuda.empty_cache()
            continue
        # Warm CUDA and disk cache before BOTH timing arms. Warm-up is not training.
        model.zero_grad(set_to_none=True)
        loss, _ = losses[0](); loss.backward(); model.zero_grad(set_to_none=True)
        runs, grads = [], []
        weights = []
        for checkpoint in [True, False]:
            model.load_state_dict(init)
            run, grad = train_pass(model, losses, checkpoint, False, stage)
            runs.append(run); grads.append(grad)
            weights.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        equal_grads = all(torch.equal(a, b) for a, b in zip(*grads))
        equal_weights = all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
        assert equal_grads and equal_weights
        results[stage] = {'baseline': runs[0], 'optimized': runs[1],
                          'gradients_exact': equal_grads, 'weights_exact': equal_weights,
                          'speedup': sum(runs[0]['seconds']) / sum(runs[1]['seconds'])}
        del source, model, ref, head, probe, init, grads, weights
        gc.collect(); torch.cuda.empty_cache()
    save('training_' + panel, {'complete': True, 'panel': panel, 'stages': results,
                               'code': hashes(), 'device': torch.cuda.get_device_name()})


def training_slots(model, probe, head, rule, source, rows, initial):
    runs, weights, gradients = [], [], []
    episodes = [p.episode(rows, 0, i) for i in range(32)]
    # Preserve policy collection/replay batch32 and checkpointed full BPTT.
    for fast in [False, True]:
        model.load_state_dict(initial)
        model.checkpoint_blocks = True
        opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                lr=3e-4, weight_decay=.05)
        secs, events, grads = [], [], []
        torch.cuda.reset_peak_memory_stats()
        context = accelerated_scoring(p)
        if fast:
            context.__enter__()
        try:
            for slot in range(16):
                def step():
                    opt.zero_grad(set_to_none=True)
                    ee, has_grad = [], False
                    for ep in episodes[2*slot:2*slot+2]:
                        loss, event = p.updater_loss(model, probe, head, rule, source, ep, verify=True)
                        ee.append(event)
                        if loss.requires_grad:
                            (loss / 2).backward(); has_grad = True
                    if has_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        opt.step()
                    return ee, has_grad
                (ee, has_grad), sec = timed(step)
                secs.append(sec); events.append(ee)
                grads.append(flat_grad(model) if has_grad else None)
                print('UPDATER_SLOT', fast, slot, sec, [e['used_updates'] for e in ee], flush=True)
        finally:
            if fast:
                context.__exit__(None, None, None)
        runs.append({'seconds': secs, 'events': events,
                     'peak_GiB': torch.cuda.max_memory_allocated()/2**30})
        gradients.append(grads)
        weights.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    equal_grads = all((a is None and b is None) or
                      (a is not None and b is not None and torch.equal(a, b))
                      for a, b in zip(*gradients))
    equal_weights = all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    assert runs[0]['events'] == runs[1]['events'] and equal_grads and equal_weights
    return {'baseline': runs[0], 'optimized': runs[1], 'gradients_exact': equal_grads,
            'weights_exact': equal_weights, 'events_exact': True,
            'speedup': sum(runs[0]['seconds']) / sum(runs[1]['seconds'])}


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob('*.py')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['inference', 'training'])
    parser.add_argument('--panel', choices=['cross', 'within'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in [MEM/'SUMMARY_FIXED.json', PROJECT/'results/adajepa_phyre_20261010/SUMMARY.json']:
        done = m.read(path)
        assert done['complete'] and done['raw_audit_passed']
    globals()[args.mode](args.panel)
