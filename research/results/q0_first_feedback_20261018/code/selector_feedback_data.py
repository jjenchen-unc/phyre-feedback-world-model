"""Q0-first policy over unchanged classifier-selected top1024 pools."""
from pathlib import Path
import sys
import numpy as np
P = Path('/path/to/phyre-unc')
R = P / 'results/q0_first_feedback_20261018'
C = P / 'results/candidate_coverage_100k_20261016'
sys.path.insert(0, str(R / 'code'))
import deploy_pool_data as data
data.R = R
import candidate_coverage_100k as coverage
coverage.R = R
read, write, sha, save = data.read, data.write, data.sha, data.save_npz


def choose_trial(scores, tried, turn):
    """Use current predictor scores at EVERY turn, including the first."""
    assert int(np.count_nonzero(tried)) == turn
    assert scores is not None and len(scores) == len(tried) and not tried.all()
    values = np.array(scores, copy=True)
    assert np.isfinite(values).all()
    values[tried] = -np.inf
    return int(values.argmax())
