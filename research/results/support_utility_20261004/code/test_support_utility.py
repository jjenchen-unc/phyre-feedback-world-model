"""CPU checks of causal pairing and data-role separation, not model quality."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
from support_utility import grouped_partition, classify, episode, ARMS


class SupportDataTests(unittest.TestCase):
    def test_cluster_transitive_overlap_is_never_split(self):
        store = SimpleNamespace(files=np.array(['a', 'b', 'c', 'd']),
            actions=np.array([[.1,.1,.1],[.2,.1,.1],[.3,.1,.1],[.8,.8,.8]]))
        roles, groups = grouped_partition(store, '00001:001', range(4), {'a':[0], 'b':[0,1], 'c':[1]})
        self.assertEqual(len({groups[str(i)]['role'] for i in range(3)}), 1)
        self.assertEqual(len({groups[str(i)]['group'] for i in range(3)}), 1)
        self.assertEqual(sorted(sum(roles.values(), [])), list(range(4)))
        self.assertEqual((roles, groups), grouped_partition(store, '00001:001', range(4), {'a':[0], 'b':[0,1], 'c':[1]}))

    def test_top20_includes_successes_and_stable_ties(self):
        # A successful top candidate consumes a high-score slot; cannot filter it out.
        labels = np.array([True,False,False,False,False,False,False,False,False,False,True])
        row = {'support':list(range(10)), 'query':[10], 'N':[3], 'far':[1,2,4,5,6,7,8,9]}
        result = classify(row, [10,9,9,8,7,6,5,4,3,2], labels)
        self.assertEqual(result['H'], [1]); self.assertIn(2, result['F'])
        self.assertTrue(result['eligible']); self.assertEqual(result['top20_support'], [0,1])

    def test_arm_changes_only_context_draw(self):
        rows = [{'task':f'{t:05d}:001', 'query':list(range(100+t*20,112+t*20)),
            'times':[[1,2,3,4,5,6,7,8]]*12, 'N':[t*10], 'H':[t*10+1], 'F':[t*10+2],
            'failures':[t*10,t*10+1,t*10+2,t*10+3]} for t in range(5)]
        categories = set()
        for seed in range(3):
            for step in range(200):
                samples = [episode(rows, seed, step, arm) for arm in ARMS]
                for s in samples[1:]: self.assertEqual(s[:3], samples[0][:3])
                self.assertIn(samples[1][3][0], samples[1][0]['N'])
                for row, ids, _, ctx, cat in samples:
                    self.assertFalse(set(ids)&set(ctx)); self.assertEqual(len(ctx),1)
                    self.assertIn(ctx[0], row[cat])
                categories.add(samples[2][4])
        self.assertEqual(categories, {'N','H','F'})

    def test_missing_category_does_not_become_eligible(self):
        row = {'support':[0,1,2], 'query':[3], 'N':[0], 'far':[1,2]}
        result = classify(row, [3,2,1], np.zeros(4,bool))
        self.assertFalse(result['eligible']); self.assertEqual(result['H'], [])


if __name__ == '__main__': unittest.main()
