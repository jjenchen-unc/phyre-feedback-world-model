"""Scoring and per-task pair AUC (not AUCCESS)."""
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
