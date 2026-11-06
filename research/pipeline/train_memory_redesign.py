"""Matched R0--R4 training with checkpointed optimizer and deterministic episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from common import MemoryPredictor
from redesign_lib import Store, load_models, context_memory, readout, write_json, digest, BASE
from redesign_sampling import sample_episode


def loss_terms(pr, truth, labels, balanced, scores=None):
    per = (pr.float()[:, 1:] - truth[:, 1:]).abs().mean((1, 2, 3))
    classes = [per[labels == y].mean() for y in [False, True] if (labels == y).any()]
    prediction = torch.stack(classes).mean() if balanced else per.mean()
    ranking = pr.sum() * 0
    if scores is not None and labels.any() and (~labels).any():
        ranking = F.softplus(-(scores[labels, None] - scores[None, ~labels]) / 4.).mean()
    return prediction, ranking, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--variant', choices=['R0', 'R1', 'R2', 'R3', 'R4'], required=True)
    p.add_argument('--steps', type=int, default=20000); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rank-weight', type=float, default=.02)
    p.add_argument('--checkpoint-every', type=int, default=2000)
    args = p.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    data = json.loads(Path(args.data).read_text()); store = Store()
    import hashlib
    assert data['files_sha256'] == hashlib.sha256('\n'.join(store.files).encode()).hexdigest()
    base, probe, rule, pc, _ = load_models()
    model = MemoryPredictor(base, pc['Din'], mode='residual', mem_steps=16, mem_grid=10,
                            per_attempt=True, gate_empty=True).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 3e-4, total_steps=args.steps,
                                                   pct_start=.03 if args.steps >= 100 else .5)
    start, counters, log = 0, Counter(), []
    state_path = out / 'resume.pt'
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        assert state['args'] == vars(args)
        model.load_state_dict(state['model'], strict=False)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        start, counters, log = state['step'], Counter(state['counters']), state['log']
    write_json(out / 'registration.json', {'args': vars(args), 'data_sha256': digest(args.data),
               'script_sha256': digest(__file__), 'base': str(BASE), 'checkpoint_hashes': data['checkpoint_hashes'],
               'episode_budget': args.steps, 'queries_per_episode': 8, 'memory_only_holdout': data['memory_holdout_templates']})
    t0 = time.time(); running = np.zeros(3); samples = 0
    for step in range(start, args.steps):
        e = sample_episode(data, store, step, args.seed, args.variant)
        counters['episodes'] += 1; counters['context_' + e['context_kind']] += 1
        counters['actual_k_' + str(len(e['context']))] += 1
        if e['context_fallback']: counters[e['context_fallback']] += 1
        for group in e['groups']: counters['query_' + group] += 1
        q = e['query']; counters['queries'] += len(q)
        y = torch.from_numpy(store.labels[q]).cuda()
        counters['positive_queries'] += int(y.sum()); counters['single_class_episodes'] += int(not (y.any() and (~y).any()))
        truth = torch.from_numpy(store.z(q)).cuda().float()
        a = torch.from_numpy(store.actions[q]).cuda()
        model.train(); base.eval(); probe.eval()
        memory, mask = context_memory(model, store, e['context'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pr = model(truth[:, 0], a, memory.expand(len(q), -1, -1), mask.expand(len(q), -1)).float()
        scores = readout(pr, [store.feature(i) for i in q], probe, rule) if args.variant == 'R4' else None
        lp, lr, per = loss_terms(pr, truth, y, args.variant in ['R2', 'R3', 'R4'], scores)
        loss = lp + (args.rank_weight * lr if args.variant == 'R4' else 0)
        if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss at {step}: {e}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(norm): raise RuntimeError(f'nonfinite gradient at {step}')
        optimizer.step(); scheduler.step()
        running += [float(lp.detach()), float(lr.detach()), float(norm)]; samples += 1
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            r = {'step': step + 1, 'prediction_loss': float(running[0]/samples),
                 'ranking_loss': float(running[1]/samples), 'grad_norm': float(running[2]/samples),
                 'seconds': round(time.time()-t0), 'variant': args.variant}
            log.append(r); print(json.dumps(r), flush=True); running[:] = 0; samples = 0
            write_json(out / 'progress.json', {'latest': r, 'counts': dict(counters)})
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            weights = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('base.')}
            checkpoint = {'state_dict': weights, 'mode': 'residual', 'mem_steps': 16, 'mem_grid': 10,
                          'per_attempt': True, 'gate_empty': True, 'unfreeze': False,
                          'base': str(BASE), 'args': vars(args), 'step': step+1, 'log': log}
            torch.save(checkpoint, out / 'memory.tmp.pt'); (out / 'memory.tmp.pt').replace(out / 'memory.pt')
            state = {'args': vars(args), 'model': weights, 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step+1,
                     'counters': dict(counters), 'log': log}
            torch.save(state, out / 'resume.tmp.pt'); (out / 'resume.tmp.pt').replace(state_path)
    write_json(out / 'training_complete.json', {'steps': args.steps, 'counts': dict(counters),
                                               'memory_sha256': digest(out / 'memory.pt')})
    print('TRAIN_DONE', args.variant, flush=True)


if __name__ == '__main__': main()
