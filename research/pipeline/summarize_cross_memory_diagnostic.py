"""Validate and plot the saved fixed-support diagnostic without model inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    names = ['base', 'empty_original', 'original_full', 'gated_full', 'gated_scale025',
             'action_outcome_only', 'no_residual', 'no_video',
             'wrong_scene_same_template', 'wrong_scene_other_template', 'observed_latents']
    reports, shifts = [], {}
    text = ['# Fixed-support diagnostic results', '',
            'Ranks below exclude the K support actions. Lower ranks/errors are better; '
            'these are frozen-ranking diagnostics, not closed-loop success rates.', '']
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
    for row, task in enumerate(['00014_273', '00014_407']):
        path = args.directory / task
        report = json.loads((path / 'summary.json').read_text())
        c = np.load(path / 'candidates.npz')
        assert len(c['indices']) == len(set(c['indices'])) == 1024
        assert report['base_reproduced'] == report['historical_base_first_success']
        arrays = {}
        for k, context in report['contexts'].items():
            excluded = [s['pack_index'] for s in context['support']]
            query = ~np.isin(c['indices'], excluded)
            assert not c['labels'][~query].any()
            text += [f'## {report["task"]}, K={k}', '',
                     '| Condition | First success rank | AUC | All latent L1 | Success latent L1 | All green error px | Success green error px |',
                     '|---|---:|---:|---:|---:|---:|---:|']
            for name in names:
                filename = name if name in ['base', 'empty_original', 'empty_gated', 'observed_latents'] else f'k{k}_{name}'
                a = dict(np.load(path / (filename + '.npz')))
                arrays[(k, name)] = a
                assert all(np.isfinite(v).all() for v in a.values())
                order = np.flatnonzero(query)[np.argsort(-a['scores'][query], kind='stable')]
                rank = int(np.flatnonzero(c['labels'][order])[0]) + 1
                m = context['conditions'][name]
                assert rank == m['first_success_rank']
                assert abs(roc_auc_score(c['labels'][query], a['scores'][query]) - m['rank_auc']) < 1e-10
                assert abs(a['latent_l1_time'][query, 1:].mean() - m['all']['latent_l1']) < 1e-7
                text.append(f'| {name} | {rank} | {m["rank_auc"]:.4f} | '
                            f'{m["all"]["latent_l1"]:.5f} | {m["successful_candidates"]["latent_l1"]:.5f} | '
                            f'{m["all"]["green_error_median_px"]:.2f} | '
                            f'{m["successful_candidates"]["green_error_median_px"]:.2f} |')
            text.append('')
        k = '1'
        summary = report['contexts'][k]['conditions']
        b, m = arrays[(k, 'base')], arrays[(k, 'gated_full')]
        q = ~np.isin(c['indices'], [s['pack_index'] for s in report['contexts'][k]['support']])
        shifts[report['task']] = {}
        for label, sel in [('success', q & c['labels']), ('failure', q & ~c['labels'])]:
            dl = m['latent_l1_time'][sel, 1:].mean(1) - b['latent_l1_time'][sel, 1:].mean(1)
            dx = m['green_xy'][sel, -1, 0] - b['green_xy'][sel, -1, 0]
            shifts[report['task']][label] = {
                'n': int(sel.sum()), 'mean_delta_latent_l1': float(dl.mean()),
                'n_latent_l1_worse': int((dl > 0).sum()),
                'mean_final_x_shift_px': float(dx.mean()), 'n_shift_right': int((dx > 0).sum()),
                'true_final_x_mean_px': float(c['green_truth'][sel, -1, 0].mean()),
                'base_final_x_mean_px': float(b['green_xy'][sel, -1, 0].mean()),
                'memory_final_x_mean_px': float(m['green_xy'][sel, -1, 0].mean())}
        bars = ['base', 'empty_original', 'gated_full', 'gated_scale025', 'action_outcome_only']
        ranks = [summary[n]['first_success_rank'] for n in bars]
        ax = axes[row, 0]
        ax.barh(['Base', 'Empty memory', 'Full feedback', 'Feedback x0.25', 'Visual channels zero'], ranks)
        ax.invert_yaxis(); ax.set_xscale('log'); ax.set_xlim(.8, 1500)
        for i, v in enumerate(ranks): ax.text(v * 1.08, i, str(v), va='center')
        ax.set_title(report['task'] + ': best success rank, K=1')
        ax.set_xlabel('Rank among remaining candidates (log scale; lower is better)')
        base = arrays[(k, 'base')]
        order = np.argsort(-base['scores'], kind='stable')
        chosen = order[np.flatnonzero(c['labels'][order])[0]]
        time = np.arange(32) * 32 / 60
        for dim, ax in enumerate(axes[row, 1:]):
            ax.plot(time, c['green_truth'][chosen, :, dim], 'k-', lw=2, label='True trajectory')
            for n, label in [('observed_latents', 'Observed-latent readout'), ('base', 'Base'),
                             ('gated_full', 'Full feedback'), ('gated_scale025', 'Feedback x0.25')]:
                ax.plot(time, arrays[(k, n)]['green_xy'][chosen, :, dim], label=label,
                        linestyle='--' if n == 'observed_latents' else '-')
            ax.set_xlabel('Simulation time (s)'); ax.set_ylabel('Green ball ' + 'xy'[dim] + ' (px)')
            ax.set_title('Base-ranked best successful action: ' + 'xy'[dim] + ' trajectory')
            ax.grid(alpha=.2)
        axes[row, 2].legend(fontsize=8)
        reports.append(report)
    assert reports[0]['sha256'] == reports[1]['sha256']
    assert reports[0]['script_sha256'] == reports[1]['script_sha256']
    text += ['Validation: base ranks reproduced; candidate IDs unique; support failures '
             'excluded; outputs finite; ranks/AUC/latent means independently recomputed; '
             'checkpoint and diagnostic script hashes identical across tasks.', '']
    (args.directory / 'METRICS.md').write_text('\n'.join(text))
    (args.directory / 'class_conditional_shifts.json').write_text(json.dumps(shifts, indent=2))
    fig.savefig(args.directory / 'diagnostic.png', dpi=160)
    fig.savefig(args.directory / 'diagnostic.pdf')
    print('\n'.join(text))


if __name__ == '__main__':
    main()
