"""Policy regression checks: predictor-first, current scores, no repeats."""
import ast
from pathlib import Path
import unittest
import numpy as np
node=next(n for n in ast.parse(Path(__file__).with_name('selector_feedback_data.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='choose_trial')
env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<policy>','exec'),env)
choose=env['choose_trial']
class PredictorFirstTests(unittest.TestCase):
    def test_first_uses_predictor_instead_of_pool_order(self):
        self.assertEqual(choose(np.array([-10.,9.,5.,0.]),np.zeros(4,bool),0),1)
        with self.assertRaises(AssertionError):choose(None,np.zeros(4,bool),0)
    def test_feedback_changes_choice_and_masks_tried(self):
        tried=np.array([False,True,False,False])
        self.assertEqual(choose(np.array([1.,100.,7.,2.]),tried,1),2)
        tried[2]=True
        self.assertEqual(choose(np.array([8.,100.,50.,9.]),tried,2),3)
    def test_q0_order_ties_and_immutable_scores(self):
        scores=np.array([1.,3.,9.,9.]);original=scores.copy();tried=np.zeros(4,bool);order=[]
        for turn in range(4):
            j=choose(scores,tried,turn);order.append(j);tried[j]=True
        self.assertEqual(order,[2,3,1,0]);np.testing.assert_array_equal(scores,original)
    def test_invalid_state_or_nonfinite_fail(self):
        for scores,tried,turn in [([1.,2.],[True,False],0),([1.,np.nan],[False,False],0),([1.,2.],[True,True],2)]:
            with self.assertRaises(AssertionError):choose(np.array(scores),np.array(tried),turn)
if __name__=='__main__':unittest.main()
