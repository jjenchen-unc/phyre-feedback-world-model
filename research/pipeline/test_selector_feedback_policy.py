"""Regression checks for externally selected first trial and causal reranking."""
import ast
from pathlib import Path
import unittest
import numpy as np

tree=ast.parse(Path(__file__).with_name('selector_feedback_data.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='choose_trial')
env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<policy>', 'exec'),env)
choose=env['choose_trial']


class SelectorFirstTests(unittest.TestCase):
    def test_first_trial_does_not_use_q0_argmax(self):
        tried=np.zeros(4,bool)
        self.assertEqual(choose(np.array([-10.,9.,5.,0.]),tried,0),0)
        self.assertEqual(choose(None,tried,0),0)

    def test_later_trial_uses_updated_scores_and_excludes_all_tried(self):
        tried=np.array([True,False,False,False])
        self.assertEqual(choose(np.array([100.,1.,7.,2.]),tried,1),2)
        tried[2]=True
        self.assertEqual(choose(np.array([100.,8.,50.,9.]),tried,2),3)

    def test_fixed_q0_scores_can_be_reused_without_mutation(self):
        score=np.array([100.,3.,9.,9.]);original=score.copy();tried=np.zeros(4,bool);order=[]
        for turn in range(4):
            j=choose(score,tried,turn);order.append(j);tried[j]=True
        self.assertEqual(order,[0,2,3,1]);np.testing.assert_array_equal(score,original)

    def test_turn_cannot_reset_after_feedback(self):
        with self.assertRaises(AssertionError):choose(None,np.array([True,False]),0)


if __name__=='__main__':unittest.main()
