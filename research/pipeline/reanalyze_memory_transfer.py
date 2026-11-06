"""Read-only endpoint-transfer measurements on two selected historical failures.

Positions are frozen-probe readouts, not decoded simulated trajectories.
This measures association and correction direction, not a causal training
mechanism or the behavior of the newly trained R0--R4 checkpoints.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cross_memory_diagnostic_20260928'


def main():
    result = {'scope': __doc__, 'tasks': {}}
    for task in ['00014_273', '00014_407']:
        p = OUT / task
        summary = json.loads((p / 'summary.json').read_text())
        with np.load(p / 'candidates.npz') as c, np.load(p / 'base.npz') as b, \
             np.load(p / 'k1_gated_full.npz') as m, \
             np.load(p / 'k1_wrong_scene_same_template.npz') as w:
            support = int(np.flatnonzero(c['indices'] == summary['support'][0]['pack_index'])[0])
            success = c['labels'].astype(bool)
            assert not success[support]
            query = np.arange(len(success)) != support
            needed = c['green_truth'][:, -1].astype(float) - b['green_xy'][:, -1]
            change = m['green_xy'][:, -1].astype(float) - b['green_xy'][:, -1]
            wrong = w['green_xy'][:, -1].astype(float) - b['green_xy'][:, -1]
            mean = change[query].mean(0)
            energy = np.square(change[query]).sum(1).mean()
            centered_energy = np.square(change[query] - mean).sum(1).mean()
            assert np.isclose(energy, centered_energy + np.square(mean).sum())
            result['tasks'][task] = {
                'n_success_queries': int(success.sum()),
                'support_needed_endpoint_correction_xy_px': needed[support].tolist(),
                'success_mean_needed_endpoint_correction_xy_px': needed[success].mean(0).tolist(),
                'success_mean_actual_endpoint_correction_xy_px': change[success].mean(0).tolist(),
                'success_fraction_negative_dot_with_needed_correction': float(
                    ((needed[success] * change[success]).sum(1) < 0).mean()),
                'all_query_mean_endpoint_correction_xy_px': mean.tolist(),
                'all_query_common_vector_energy_fraction': float(np.square(mean).sum() / energy),
                'correct_wrong_scene_endpoint_correction_cosine': float(
                    (change[query] * wrong[query]).sum() /
                    np.sqrt(np.square(change[query]).sum() * np.square(wrong[query]).sum())),
            }
    path = OUT / 'transfer_direction_reanalysis.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['tasks'], indent=2))


if __name__ == '__main__':
    main()
