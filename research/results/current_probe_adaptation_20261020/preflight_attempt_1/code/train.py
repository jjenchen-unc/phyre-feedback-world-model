"""Matched target draws; adapt ranking probe only."""
from adapt_runtime import *
import argparse
import numpy as np


def train(arm, smoke=False):
    import torch
    import torch.nn.functional as F
    from common import AttnProbe
    plan = checked()
    if smoke:
        task = plan['fit'][0]
        entries = read(R/'cache'/task.replace(':', '_')/'complete.json')['states']
    else:
        audit = read(R/'DATA_AUDIT.json')
        assert audit['complete'] and audit['plan'] == sha(R/'PLAN.json')
        entries = audit['entries']['fit']
    records, grouped = [], {}
    for relative in entries:
        folder = R/relative
        meta = read(folder/'metadata.json')
        i = len(records)
        records.append(dict(meta=meta, **{k: np.load(folder/(k+'.npy'), mmap_mode='r') for k in ['predicted', 'observed', 'features']}))
        grouped.setdefault(meta['task'], {}).setdefault(meta['variant'], []).append(i)
    tasks = sorted(grouped)
    ck = torch.load(plan['initial_head'], map_location='cpu', weights_only=False)
    torch.manual_seed(plan['seed'])
    probe = AttnProbe(ck['D'], ck['S']).cuda()
    probe.load_state_dict(ck['state_dict'])
    probe.train().requires_grad_(True)
    initial = lr.source().old.ident(probe)
    opt = torch.optim.AdamW(probe.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    steps = 3 if smoke else plan['steps']
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(plan['seed'])
    draws = hashlib.sha256()
    log = []
    start = time.time()
    out = R/('smoke_heads' if smoke else 'heads')/arm
    assert not (out/'complete.json').exists(), 'Refuse duplicate completed training'
    for step in range(steps):
        tokens, queries, targets, angles, bars, indices = [], [], [], [], [], []
        for j in range(plan['batch']):
            task = tasks[rng.integers(len(tasks))]
            variants = sorted(grouped[task])
            variant = variants[rng.integers(len(variants))]
            i = int(rng.choice(grouped[task][variant]))
            row = records[i]
            query = int(rng.integers(8))
            slot = int(rng.integers(8))
            f = row['features'][query]
            dynamic = np.flatnonzero((f[0, :, 13] < .5) & (f[0, :, 11] < .5))
            obj = int(rng.choice(dynamic))
            t = row['meta']['times'][slot]
            source = 'predicted' if arm == 'mixed' and (j+step) % 2 else 'observed'
            tokens.append(row[source][query, slot])
            queries.append(f[0, obj])
            targets.append(f[t, obj, :2]*256)
            angle = f[t, obj, 2]*4*np.pi
            angles.append([np.cos(angle), np.sin(angle)])
            bars.append(f[0, obj, 4:8].argmax() == 1)
            indices.append((i, query, slot, obj))
        draws.update(np.asarray(indices, np.int64).tobytes())
        tok = torch.from_numpy(np.stack(tokens)).cuda().float()
        q = torch.from_numpy(np.stack(queries)).cuda().float()
        xy, angle = probe.read(probe.inp(tok)+probe.space, q)
        target = torch.tensor(np.array(targets), device='cuda', dtype=torch.float32)
        ang = torch.tensor(angles, device='cuda', dtype=torch.float32)
        bar = torch.tensor(bars, device='cuda', dtype=torch.float32)
        loss = 10*F.l1_loss(xy, target)/256 + ((angle-ang).abs().mean(-1)*bar).mean()
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), plan['clip'])
        assert torch.isfinite(norm) and norm > 0
        opt.step()
        sched.step()
        if (step+1) % 100 == 0 or step+1 == steps:
            record = dict(step=step+1, loss=float(loss.detach()), gradient_norm=float(norm), seconds=time.time()-start)
            log.append(record)
            write(out/'progress.json', record)
            print('TRAIN', arm, record, flush=True)
    assert lr.source().old.ident(probe) != initial
    out.mkdir(parents=True, exist_ok=True)
    tmp = out/'probe.tmp'
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in probe.state_dict().items()},
                    D=ck['D'], S=ck['S'], rule=ck['rule']), tmp)
    tmp.replace(out/'probe.pt')
    write(out/'complete.json', dict(complete=True, plan=sha(R/'PLAN.json'), steps=steps,
                                   checkpoint_sha256=sha(out/'probe.pt'), draw_sha256=draws.hexdigest(),
                                   initial_sha256=initial, arm=arm, log=log))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['true_control', 'mixed'], required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    train(a.arm, a.smoke)
