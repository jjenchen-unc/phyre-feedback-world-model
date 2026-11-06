from direction_runtime import *
import argparse


def geometry(h, tokens, features, times):
    import torch
    from common import surface_dist
    items = {'all': [], 'task': [], 'other': [], 'closest_sampled_time': []}
    for z, f in zip(tokens, features):
        kv = h.inp(torch.from_numpy(np.array(z)).cuda().float())+h.space
        closest = int(np.argmin(surface_dist(f[times])))
        for obj in np.flatnonzero((f[0, :, 13] < .5) & (f[0, :, 11] < .5)):
            q = torch.from_numpy(np.repeat(f[:1, obj], len(times), axis=0)).cuda().float()
            xy, ang = h.read(kv, q)
            targetxy = torch.from_numpy(f[times, obj, :2].copy()).cuda().float()*256
            angle = torch.from_numpy(f[times, obj, 2].copy()).cuda().float()*4*np.pi
            targetangle = torch.stack([angle.cos(), angle.sin()], -1)
            loss = 10*(xy-targetxy).abs().mean(-1)/256
            if f[0, obj, 4:8].argmax() == 1:
                loss = loss+(ang-targetangle).abs().mean(-1)
            relevant = bool(f[0, obj, 9] > .5 or f[0, obj, 10] > .5)
            value = float(loss.mean()); items['all'].append(value); items['task' if relevant else 'other'].append(value)
            if relevant: items['closest_sampled_time'].append(float(loss[closest]))
    return {k: float(np.mean(v)) if v else None for k,v in items.items()}


def run(stage, shard, nshards):
    import torch
    from fast_inference import PreparedScorer
    plan = checked(); branch = read(R/'ROUND1.json'); selected = branch['selected_variant']; zero = selected.split('_')[0]+'_zero'
    smoke = stage == 'smoke'
    if smoke:
        tasks = [next(t for t in plan['fit'] if (folder(t, selected, 1)/'complete.json').exists())]
    else:
        assert read(R/'TRAIN_AUDIT.json')['passed']; tasks = plan[stage][shard::nshards]
    arms = ARMS[1:]
    if stage == 'confirmation':
        decision = read(R/'ROUND2_DECISION.json')
        arms = sorted(set(['memory_mixed', decision['selected_memory_arm']]))
    head_names = ['old', 'previous_mixed']+[a+'_'+str(s) for a in ['q0_mixed']+arms for s in SEEDS]
    heads = {name: load_head(name, smoke) for name in head_names}
    scorers = {name: PreparedScorer(h, ['soft', 4, 1]) for name,h in heads.items()}
    team = ar.make_team(); assertions = []
    try:
        for task in tasks:
            out = R/stage/task.replace(':', '_'); team.load(task)
            qd = folder(task, 'q0', 0); qfirst = int(np.load(qd/'scores.npy').argmax())
            failed = bool(team.pool['status'][qfirst] < 0)
            variants = ['q0', selected, zero] if failed else ['q0']
            records = []
            for variant in variants:
                depth = 0 if variant == 'q0' else 1; d = folder(task, variant, depth)
                meta = read(d/'metadata.json')
                if depth: assert meta['tried'] == [qfirst]
                ar.configure(team, variant); fingerprint = ar.lr.source().old.ident(team.model)
                state = torch.from_numpy(np.load(d/'state.npy')).cuda()
                names = ['old', 'previous_mixed']+[a+'_'+str(s) for a in (['q0_mixed'] if variant == 'q0' else arms) for s in SEEDS]
                scores = {name: [] for name in names}
                with torch.no_grad():
                    for start in range(0, 1024, 32):
                        z, a = team.chunk(start)
                        with torch.autocast('cuda', dtype=torch.bfloat16): pred = team.model(z, a, state.expand(32,-1,-1))
                        fs = [f[None] for f in team.pool['features'][start:start+32]]
                        for name in names: scores[name].append(scorers[name](pred, fs).cpu().numpy())
                    scores = {k:np.concatenate(v) for k,v in scores.items()}
                    np.testing.assert_array_equal(scores['old'], np.load(d/'scores.npy'))
                    # Old mixed readout also has a recorded full ranking on the validation scenes.
                    previous = A/'fixed'/task.replace(':','_')/f'{variant}_{depth}.npz'
                    if previous.exists():
                        with np.load(previous) as ref: np.testing.assert_array_equal(scores['previous_mixed'], ref['mixed'])
                    if smoke and variant == 'q0':
                        initial = [np.repeat(f[None],32,axis=0) for f in team.pool['features'][-32:]]
                        for name in names:
                            first = scorers[name](pred,initial).cpu().numpy()
                            changed = [f.copy() for f in initial]
                            for f in changed:f[1:] = 123
                            np.testing.assert_array_equal(first,scorers[name](pred,changed).cpu().numpy())
                    cached = {s:np.load(d/(s+'.npy')) for s in ['predicted','observed','features']}
                    headmetrics = {}
                    for name in names:
                        headmetrics[name] = dict(rank=rank_metrics(scores[name],team.pool['status']>0,meta['tried']),
                                                 after_first_failure=rank_metrics(scores[name],team.pool['status']>0,[qfirst]) if failed else None,
                                                 geometry={s:geometry(heads[name],cached[s],cached['features'],meta['times']) for s in ['predicted','observed']})
                assert ar.lr.source().old.ident(team.model)==fingerprint
                ar.lr.source().data.save_npz(out/(variant+'.npz'),**scores,labels=team.pool['status']>0,indices=team.pool['action_idx'],tried=np.array(meta['tried'],np.int64),qfirst=np.array(qfirst))
                records.append(dict(variant=variant,depth=depth,heads=headmetrics))
            write(out/'result.json',dict(complete=True,plan=sha(R/'PLAN.json'),task=task,first_failed=failed,old_exact=True,model_unchanged=True,records=records))
            print('EVAL',stage,task,variants,flush=True)
        if smoke:write(R/'SMOKE_EVAL.json',dict(passed=True,plan=sha(R/'PLAN.json'),old_scores_exact=True,future_input_invariant=True,model_unchanged=True))
    finally: ar.lr.close(team)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','discovery','confirmation']);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=8);a=p.parse_args();run(a.stage,a.shard,a.nshards)
