import os
os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
import sys
from pathlib import Path
import numpy as np
import torch
P = Path('/path/to/phyre-unc')
R = P/'results/current_probe_adaptation_20261020'
sys.path.insert(0, str(R/'code'))
import adapt_runtime as rt
import selector_feedback_run as src
torch.set_num_threads(2)
source = src.data.checked()
cov = src.pools.coverage
model = cov.scene_model('cross', source)
with np.load(src.pools.C/'inputs/cross_tokens.npz') as z:
    token = z['tokens'][0]
actions = np.load(R/'inputs/actions.npy')
actual = cov.scene_scores(model, token, actions)
task = source['panels']['cross']['tasks'][0]
with np.load(src.pools.C/'scores/cross/scene'/task.replace(':', '_')/'scores.npz') as z:
    expected = z['scores'].copy()
rt.write(R/'CLASSIFIER_NUMERICS.json', dict(
    exact=bool(np.array_equal(actual, expected)), max_error=float(np.abs(actual-expected).max()),
    mismatched=int((actual != expected).sum()), without_workspace_override=True))
np.testing.assert_array_equal(actual, expected)
print('CLASSIFIER_100K_EXACT_WITHOUT_WORKSPACE_OVERRIDE', flush=True)
