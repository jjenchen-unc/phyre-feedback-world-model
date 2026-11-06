import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from train import sample_query
class SamplingTests(unittest.TestCase):
 def test_uniform_preserves_original_sampler(self):
  perm=np.random.default_rng(7).permutation(32);selected=[int(perm[1]),int(perm[5])]
  self.assertEqual(sample_query(perm,selected,np.ones(32),'uniform'),[int(x) for x in perm if x not in selected][:8])
 def test_balanced_labels_and_permutation_order(self):
  perm=np.random.default_rng(7).permutation(32);labels=np.array([1]*16+[-1]*16);selected=[int(perm[0])]
  q=sample_query(perm,selected,labels,'balanced');self.assertEqual(sum(labels[q]>0),4);self.assertEqual(sum(labels[q]<0),4);self.assertFalse(set(q)&set(selected));self.assertEqual(q,[int(x) for x in perm if x in q])
 def test_shortage_fills_without_duplicates(self):
  for positives in [0,1,2,31,32]:
   labels=np.array([1]*positives+[-1]*(32-positives));q=sample_query(np.arange(32),[0],labels,'balanced');self.assertEqual(len(set(q)),8);self.assertNotIn(0,q)
if __name__=='__main__':unittest.main()
