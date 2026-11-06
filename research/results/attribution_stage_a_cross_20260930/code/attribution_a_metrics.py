"""Pure NumPy metrics for fixed-query attribution; geometry is a proxy, not a simulator label."""
import numpy as np


def log_score(distance, rule, weighted=False):
    margin = (float(rule[1]) - np.asarray(distance, np.float64)) / float(rule[2])
    terms = -np.logaddexp(0., -margin)
    if weighted:
        # Trapezoidal mean over a common physical interval, scaled to 32 samples.
        weights = np.ones(terms.shape[-1]); weights[[0, -1]] = .5
        weights *= 32. / (terms.shape[-1] - 1)
        terms += np.log(weights)
    return np.logaddexp.reduce(terms, axis=-1)


def auc(labels, scores):
    labels = np.asarray(labels, bool)
    if not labels.any() or labels.all():
        return None
    diff = scores[labels, None] - scores[None, ~labels]
    return float(((diff > 0) + .5 * (diff == 0)).mean())


def describe(data, truth, labels, query_mask, rule, green, dynamic_goals, base=None):
    mask = np.asarray(query_mask, bool); y = np.asarray(labels, bool)[mask]
    d = data['distance'][mask]; td = truth['distance'][mask]
    xy = data['xy'][mask]; txy = truth['xy'][mask]
    scores = data['log_score'][mask]
    ranked = y[np.argsort(-scores, kind='stable')]
    rank = int(np.flatnonzero(ranked)[0]) + 1 if y.any() else None
    out = {'n_query': len(y), 'n_success': int(y.sum()), 'auc': auc(y, scores),
           'first_success_rank': rank, 'hit_at_10': bool(ranked[:10].any()),
           'tie_fraction': 1. - len(np.unique(scores)) / len(scores),
           'margin_mean_success_minus_failure': float(scores[y].mean() - scores[~y].mean()) if y.any() and (~y).any() else None}
    per = {'latent_l1': data['latent_l1'][mask], 'latent_l2_rms': data['latent_l2_rms'][mask],
           'surface_mae': np.abs(d[:, 1:] - td[:, 1:]).mean(1),
           'green_position_error': np.linalg.norm(xy[:, 1:, green] - txy[:, 1:, green], axis=-1).mean(1)}
    near_truth = td[:, 1:] <= float(rule[1])
    count = near_truth.sum(1)
    per['surface_mae_at_true_near_contact'] = np.divide((np.abs(d[:, 1:] - td[:, 1:]) * near_truth).sum(1), count,
        out=np.full(len(y), np.nan), where=count > 0)
    if dynamic_goals:
        per['dynamic_goal_position_error'] = np.linalg.norm(xy[:, 1:, dynamic_goals] - txy[:, 1:, dynamic_goals], axis=-1).mean((1, 2))
    for threshold, name in [(0., 'surface_overlap'), (float(rule[1]), 'near_contact_tau')]:
        near, true_near = d[:, 1:] <= threshold, td[:, 1:] <= threshold
        per[name + '_time_disagreement'] = (near != true_near).mean(1)
        per[name + '_duration_error_steps'] = np.abs(near.sum(1) - true_near.sum(1)) * 32.
        def longest_run(arr):
            current = np.zeros(len(arr), int); longest = current.copy()
            for column in arr.T:
                current = (current + 1) * column; longest = np.maximum(longest, current)
            return longest
        per[name + '_longest_run_error_steps'] = np.abs(longest_run(near) - longest_run(true_near)) * 32.
        first = np.where(near.any(1), (near.argmax(1) + 1) * 32., np.nan)
        tf = np.where(true_near.any(1), (true_near.argmax(1) + 1) * 32., np.nan)
        per[name + '_first_time_error_steps_when_both'] = np.abs(first - tf)
        per[name + '_ever_disagreement'] = (near.any(1) != true_near.any(1)).astype(float)
    for group, gm in [('all', np.ones(len(y), bool)), ('success', y), ('failure', ~y)]:
        out[group] = {}
        for key, values in per.items():
            good = gm & np.isfinite(values)
            out[group][key] = float(values[good].mean()) if good.any() else None
            out[group][key + '_n'] = int(good.sum())
    if base is not None and y.any() and (~y).any():
        old = base['log_score'][mask]
        before = old[y, None] - old[None, ~y]
        after = scores[y, None] - scores[None, ~y]
        out['pair_correct_to_wrong_fraction'] = float(((before > 0) & (after < 0)).mean())
        out['pair_wrong_to_correct_fraction'] = float(((before < 0) & (after > 0)).mean())
        # Ties retain half credit in AUC; strict flips alone are not total AUC change.
    return out
