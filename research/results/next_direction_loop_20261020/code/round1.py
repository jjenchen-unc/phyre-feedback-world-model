from direction_runtime import *


def run():
    plan = checked()
    variants = ['l1_raw', 'l1_centered', 'smooth_raw', 'smooth_centered']
    reports = {}
    for split in ['fit', 'discovery', 'confirmation']:
        tasks = plan[split]
        rows = []
        for task in tasks:
            with np.load(A/'pools'/task.replace(':', '_')/'pool.npz') as z:
                labels = z['status'] > 0
            q0 = np.load(folder(task, 'q0', 0)/'scores.npy')
            first = int(q0.argmax())
            if labels[first]:
                assert all(not (folder(task, v, 1)/'complete.json').exists() for v in variants)
                continue
            base = rank_metrics(q0, labels, [first])
            for variant in variants:
                d = folder(task, variant, 1)
                zero = variant.split('_')[0]+'_zero'
                zd = folder(task, zero, 1)
                m, zm = read(d/'metadata.json'), read(zd/'metadata.json')
                assert m['tried'] == zm['tried'] == [first]
                rows.append(dict(task=task, variant=variant, q0=base,
                                 memory=rank_metrics(np.load(d/'scores.npy'), labels, [first]),
                                 zero=rank_metrics(np.load(zd/'scores.npy'), labels, [first])))
        summary = {}
        for v in variants:
            rs = [r for r in rows if r['variant'] == v]
            summary[v] = dict(n=len(rs), **{s: mean_metrics([r[s] for r in rs]) for s in ['q0', 'memory', 'zero']})
        reports[split] = dict(total_scenes=len(tasks), eligible_scenes=len({r['task'] for r in rows}), rows=rows, summary=summary)
    # The variant decision reads FIT ONLY. Validation is descriptive until the registered head selection.
    fit = reports['fit']['summary']
    def key(v):
        m = fit[v]
        return (m['memory']['top8']-m['zero']['top8'], m['memory']['top8']-m['q0']['top8'],
                m['memory']['top1']-m['zero']['top1'], (m['memory']['auc'] or 0)-(m['zero']['auc'] or 0))
    selected = max(variants, key=key)
    train_tasks = {t for t in plan['fit'] if (folder(t, selected, 1)/'complete.json').exists()}
    audit = read(A/'DATA_AUDIT.json')
    entries = [x for x in audit['entries']['fit'] if read(A/x/'metadata.json')['variant'] == selected]
    counts = dict(scenes=len(train_tasks), states=len(entries), query_instances=8*len(entries),
                  positive_query_instances=sum(sum(y > 0 for y in read(A/x/'metadata.json')['status']) for x in entries))
    assert counts['scenes'] >= 8 and counts['states'] >= 8
    write(R/'ROUND1.json', dict(complete=True, plan=sha(R/'PLAN.json'), selected_variant=selected,
                               fit_selection_keys={v: key(v) for v in variants}, training_counts=counts,
                               panels=reports, first_failure_history_exact=True))
    print('ROUND1', selected, counts, fit, flush=True)


if __name__ == '__main__':
    run()
