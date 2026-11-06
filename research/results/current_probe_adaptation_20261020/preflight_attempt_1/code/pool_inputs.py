"""Run with PHYRE CPU Python; no model-dependent scene sampling."""
import json
from pathlib import Path
import numpy as np
import phyre

P = Path(__file__).resolve().parents[3]
R = P/'results/current_probe_adaptation_20261020'
plan = json.loads((R/'PLAN.json').read_text())
tasks = plan['fit']+plan['validation']
train, dev, test = phyre.get_fold('ball_cross_template', 1)
assert set(tasks) <= set(train)
sim = phyre.initialize_simulator(tasks, 'ball')
out = R/'inputs'
out.mkdir(exist_ok=True)
np.savez(out/'scenes.npz', tasks=np.array(tasks), frames=np.array(sim.initial_scenes))
cache = phyre.get_default_100k_cache('ball')
np.savez(out/'validity.npz', tasks=np.array(tasks),
         valid=np.stack([cache.load_simulation_states(t) != 0 for t in tasks]))
np.save(out/'actions.npy', cache.action_array)
with np.load(P/'results/candidate_coverage_100k_20261016/inputs/cross_tokens.npz') as z:
    reference_tasks = z['tasks'][:32].tolist()
reference_sim = phyre.initialize_simulator(reference_tasks, 'ball')
np.save(out/'reference_frames.npy', np.array(reference_sim.initial_scenes))
print('OFFICIAL_TRAIN_INPUTS', len(tasks), flush=True)
