"""Read-only inventory of reusable trajectories; never constructs a training split."""
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/path/to/phyre-unc')
ASSETS = Path('/mnt/meg/jjenchen/phyre-unc')
OUT = ROOT/'results/multiturn_validation_20261014/data_reuse.json'
assert not OUT.exists()
sys.argv = ['make_rollouts.py', '--n-success', '16', '--n-fail', '0',
            '--n-random', '16', '--diverse']
generator = runpy.run_path(str(ROOT/'pipeline/make_rollouts.py'), run_name='inventory')
args, pick = generator['args'], generator['pick_actions']
panels = [('cross', 0), ('cross', 2), ('within', 1), ('within', 2)]
tasksets = {f'{p}{f}': set(generator['phyre'].get_fold(f'ball_{p}_template', f)[0])
            for p, f in panels}
all_tasks = set.union(*tasksets.values())
result = {'complete': False, 'panels': {}, 'recipes': {},
          'scope': 'Immutable raw/latent reuse inventory; fold-specific models and heads still required.'}
all_needed, all_existing = set(), set()
for recipe in ['s16r16', 'clusters']:
    args.clusters = 0 if recipe == 's16r16' else 4
    by_task = defaultdict(list)
    duplicates = 0
    pack_keys = set()
    for panel, fold in [('cross', 1), ('within', 0)]:
        rows = [json.loads(s) for s in (ROOT/'data/manifests'/f'ball_{panel}_template_f{fold}_train_{recipe}.jsonl').read_text().splitlines()]
        groups = defaultdict(list)
        for row in rows:
            groups[row['task']].append((row['action_idx'], row.get('cluster')))
        for task, values in groups.items():
            if task in by_task:
                assert by_task[task] == values, (recipe, task)
                duplicates += 1
            by_task[task] = values
        prefix = ASSETS/'packs'/f'{panel}_f{fold}_train_{recipe}_pca128'
        with np.load(str(prefix)+'_meta.npz') as meta:
            files = [str(x) for x in meta['files']]
            tasks = [str(x) for x in meta['tasks']]
        assert len(files) == len(rows)
        assert all(Path(f).name == Path(r['file']).name and t == r['task']
                   for f, t, r in zip(files, tasks, rows))
        latents = np.load(str(prefix)+'_latents.npy', mmap_mode='r')
        assert len(latents) == len(rows) and latents.shape[1:] == (32, 196, 128)
        pack_keys.update((r['task'], r['action_idx']) for r in rows)
    covered = all_tasks & set(by_task)
    missing = sorted(all_tasks - set(by_task))
    # Check the current generator against real, overlapping manifests before reuse.
    for task in sorted(covered)[::max(1, len(covered)//12)]:
        values = [(v[0], v[2] if len(v) > 2 else None) for v in pick(task)]
        assert values == by_task[task], ('generator changed', recipe, task)
    for i, task in enumerate(missing):
        by_task[task] = [(v[0], v[2] if len(v) > 2 else None) for v in pick(task)]
        if i % 50 == 0: print(recipe, i, '/', len(missing), flush=True)
    required = {(t, ai) for t in all_tasks for ai, _ in by_task[t]}
    result['recipes'][recipe] = {
        'shared_task_manifests_exact': duplicates,
        'new_union_tasks': len(all_tasks), 'tasks_with_existing_recipe': len(covered),
        'tasks_generated_readonly': len(missing), 'required_unique_actions': len(required),
        'in_existing_packs': len(required & pack_keys),
        'missing_from_recipe_packs': len(required - pack_keys),
    }
    for name, tasks in tasksets.items():
        result['panels'].setdefault(name, {})[recipe] = {
            'tasks': len(tasks), 'rows': sum(len(by_task[t]) for t in tasks),
            'tasks_with_existing_recipe': len(tasks & covered)}
    all_needed |= required
    all_existing |= pack_keys
    OUT.write_text(json.dumps(result, indent=2)+'\n')
new = sorted(all_needed - all_existing)
result['union'] = {'required_unique_actions': len(all_needed),
                   'present_in_existing_packs': len(all_needed & all_existing),
                   'absent_from_existing_packs': len(new),
                   'missing_keys': new}
# A standalone immutable latent may exist even when absent from the four packs.
latent_dir = ASSETS/'latents/levjepa_overlap_g32_pca128'
existing_latents = [k for k in new if (latent_dir/f'{k[0].replace(":", "_")}_{k[1]:06d}.npy').is_file()]
existing_raw = [k for k in new if (ROOT/'data/rollouts'/f'{k[0].replace(":", "_")}_{k[1]:06d}.npz').is_file()]
result['union'].update(standalone_latent_files=len(existing_latents), raw_files=len(existing_raw),
                       files_needing_encoding=len(new)-len(existing_latents))
result['complete'] = True
OUT.write_text(json.dumps(result, indent=2)+'\n')
print({k:v for k,v in result['union'].items() if k!='missing_keys'}, flush=True)
