from direction_runtime import *
import argparse


def train(index, smoke=False):
    import torch
    from common import AttnProbe
    plan = checked(); branch = read(R/'ROUND1.json')
    assert branch['complete'] and branch['plan'] == sha(R/'PLAN.json')
    arm, seed = ARMS[index//2], SEEDS[index%2]
    variant = 'q0' if arm == 'q0_mixed' else branch['selected_variant']
    entries = read(A/'DATA_AUDIT.json')['entries']['fit']
    entries = [x for x in entries if read(A/x/'metadata.json')['variant'] == variant]
    rows, by_task = [], {}
    for relative in entries:
        d = A/relative; meta = read(d/'metadata.json'); receipt = read(d/'complete.json')
        assert receipt['plan'] == sha(A/'PLAN.json')
        for name in ['metadata.json', 'predicted.npy', 'observed.npy', 'features.npy']:
            assert sha(d/name) == receipt['hashes'][name]
        rows.append(dict(meta=meta, **{s: np.load(d/(s+'.npy'), mmap_mode='r') for s in ['predicted', 'observed', 'features']}))
        by_task.setdefault(meta['task'], []).append(len(rows)-1)
    tasks = sorted(by_task)
    assert set(tasks) <= set(plan['fit']) and not set(tasks) & set(plan['discovery']+plan['confirmation'])
    ck = torch.load(plan['initial_head'], map_location='cpu', weights_only=False)
    ar.lr.source(); torch.manual_seed(seed)
    h = AttnProbe(ck['D'], ck['S']).cuda(); h.load_state_dict(ck['state_dict']); h.train()
    initial = ar.lr.source().old.ident(h)
    steps = 3 if smoke else plan['steps']
    out = R/('smoke_heads' if smoke else 'heads')/(arm+'_'+str(seed))
    assert not (out/'complete.json').exists(), 'Do not overwrite a trained head'
    opt = torch.optim.AdamW(h.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed); context_hash = hashlib.sha256(); draw_hash = hashlib.sha256()
    started = time.time(); log = []
    for step in range(steps):
        tokens, queries, xy, angle, bar, context, draws = [], [], [], [], [], [], []
        for j in range(plan['batch']):
            task = tasks[rng.integers(len(tasks))]
            i = int(rng.choice(by_task[task])); qi = int(rng.integers(8)); slot = int(rng.integers(8)); u = float(rng.random())
            row = rows[i]; f = row['features'][qi]; t = row['meta']['times'][slot]
            dynamic = np.flatnonzero((f[0, :, 13] < .5) & (f[0, :, 11] < .5))
            if arm == 'memory_task_objects':
                dynamic = dynamic[(f[0, dynamic, 9] > .5) | (f[0, dynamic, 10] > .5)]
            assert len(dynamic)
            obj = int(dynamic[min(int(u*len(dynamic)), len(dynamic)-1)])
            source = 'predicted' if arm == 'memory_predicted' or (j+step)%2 else 'observed'
            tokens.append(row[source][qi, slot]); queries.append(f[0, obj]); xy.append(f[t, obj, :2]*256)
            a = f[t, obj, 2]*4*np.pi; angle.append([np.cos(a), np.sin(a)]); bar.append(f[0, obj, 4:8].argmax() == 1)
            context.append((i, qi, slot)); draws.append((i, qi, slot, obj))
        context_hash.update(np.array(context, np.int64).tobytes()); draw_hash.update(np.array(draws, np.int64).tobytes())
        z = torch.from_numpy(np.stack(tokens)).cuda().float(); q = torch.from_numpy(np.stack(queries)).cuda().float()
        predxy, predangle = h.read(h.inp(z)+h.space, q)
        target = torch.tensor(np.array(xy), dtype=torch.float32, device='cuda')
        targetangle = torch.tensor(angle, dtype=torch.float32, device='cuda'); bars = torch.tensor(bar, dtype=torch.float32, device='cuda')
        loss = 10*(predxy-target).abs().mean()/256 + ((predangle-targetangle).abs().mean(-1)*bars).mean()
        assert torch.isfinite(loss); opt.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(h.parameters(), plan['clip']); assert torch.isfinite(norm) and norm > 0
        opt.step(); sched.step()
        if (step+1)%200 == 0 or step+1 == steps:
            rec = dict(step=step+1, loss=float(loss.detach()), grad_norm=float(norm), seconds=time.time()-started)
            log.append(rec); write(out/'progress.json', rec); print('TRAIN', arm, seed, rec, flush=True)
    assert ar.lr.source().old.ident(h) != initial
    out.mkdir(parents=True, exist_ok=True)
    tmp = out/'probe.tmp'; torch.save(dict(D=ck['D'], S=ck['S'], rule=ck['rule'], state_dict={k:v.detach().cpu() for k,v in h.state_dict().items()}), tmp); tmp.replace(out/'probe.pt')
    write(out/'complete.json', dict(complete=True, plan=sha(R/'PLAN.json'), arm=arm, seed=seed,
                                   variant=variant, steps=steps, tasks=tasks, initial_sha256=initial,
                                   context_draw_sha256=context_hash.hexdigest(), target_draw_sha256=draw_hash.hexdigest(),
                                   checkpoint_sha256=sha(out/'probe.pt'), seconds=time.time()-started, log=log))


def audit(smoke=False):
    plan = checked(); root = R/('smoke_heads' if smoke else 'heads'); rows = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = arm+'_'+str(seed); rec = read(root/name/'complete.json')
            assert rec['complete'] and rec['plan'] == sha(R/'PLAN.json') and rec['steps'] == (3 if smoke else 2000)
            assert sha(root/name/'probe.pt') == rec['checkpoint_sha256']; rows[name] = rec
    assert len({r['initial_sha256'] for r in rows.values()}) == 1
    for seed in SEEDS:
        group = [rows[arm+'_'+str(seed)] for arm in ARMS[1:]]
        assert len({r['context_draw_sha256'] for r in group}) == 1
        assert group[0]['target_draw_sha256'] == group[1]['target_draw_sha256']
    write(R/('SMOKE_TRAIN_AUDIT.json' if smoke else 'TRAIN_AUDIT.json'), dict(passed=True, plan=sha(R/'PLAN.json'), shared_initialization=True, matched_memory_context_draws=True, matched_mixed_predicted_targets=True))
    print('HEAD_AUDIT_PASSED', smoke, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, default=0); p.add_argument('--smoke', action='store_true'); p.add_argument('--audit', action='store_true'); a = p.parse_args()
    audit(a.smoke) if a.audit else train(a.index, a.smoke)
