"""Real 1024-score replay, input boundary, and unchanged predictor checks."""
from adapt_runtime import *
import numpy as np


def main():
    import torch
    from fast_inference import PreparedScorer
    plan = checked()
    team = make_team()
    configure(team, 'l1_raw')
    task = plan['test'][0]
    team.load(task)
    fingerprint = lr.source().old.ident(team.model)
    state = team.model.initial_state().detach()
    with torch.no_grad():
        scores = team.scores(state, verify=True)
        with np.load(lr.S/'evaluation/cross/q0'/task.replace(':', '_')/'scores.npz') as ref:
            np.testing.assert_array_equal(scores, ref['scores'][0])
            np.testing.assert_array_equal(team.pool['action_idx'], ref['indices'])
        pred = predict(team, state, list(range(32)))
        fs = [np.repeat(f[None], 32, axis=0) for f in team.pool['features'][:32]]
        scorer = PreparedScorer(team.head, team.rule)
        a = scorer(pred, fs).cpu().numpy()
        for f in fs:
            f[1:] = 123
        b = scorer(pred, fs).cpu().numpy()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, scores[:32])
    assert lr.source().old.ident(team.model) == fingerprint
    lr.close(team)
    write(R/'PREFLIGHT.json', dict(passed=True, task=task, old_scores_exact=True,
                                  future_input_invariant=True, original_batch32_exact=True,
                                  model_unchanged=True, plan=sha(R/'PLAN.json')))
    print('PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    main()
