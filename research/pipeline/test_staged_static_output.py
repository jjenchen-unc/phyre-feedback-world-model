"""Regression: first static evaluation writes into a previously nonexistent folder."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import query_token_staged as s


class Source:
    lat = [np.zeros((3, 2, 4, 8), np.float32)]
    actions = np.zeros((3, 3), np.float32)
    labels = np.array([False, True, False])

    def feature(self, index):
        return np.zeros((2, 1, 14), np.float32)


def check(precreate=False):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)/'new'/'closed'/'task'
        if precreate:
            folder.mkdir(parents=True)
        values = [np.array([3., 2., 1.]), np.array([1., 3., 2.])]
        with patch.object(s.q, 'scores', side_effect=values):
            result = s.static_case(folder, {'indices':[0, 1, 2], 'task':'synthetic'},
                Source(), torch.nn.Identity(), None, None, None, {'test':True})
        assert result['original_first_rank'] == 2 and result['q0_first_rank'] == 1
        with np.load(folder/'scores.npz') as saved:
            np.testing.assert_array_equal(saved['q0'], values[1])
        with patch.object(s.q, 'scores', side_effect=AssertionError('must resume without inference')):
            assert s.static_case(folder, {}, None, None, None, None, None, {'test':True}) == result


if __name__ == '__main__':
    check()
    print('STATIC_OUTPUT_REGRESSION_PASS')
