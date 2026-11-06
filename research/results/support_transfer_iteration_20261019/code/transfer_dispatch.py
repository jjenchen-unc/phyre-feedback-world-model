from transfer_core import *
import argparse

def run(stage,index):
 panels=['cross','within']
 if stage=='smoke':
  from transfer_policy import smoke
  smoke(panels[index]);return
 if stage in ['first','history']:
  from transfer_probe import first,history
  (first if stage=='first' else history)(panels[index%2],index//2,4);return
 assert stage=='policies'
 from transfer_policy import run as policy
 arm=index%8;mode=['norm_matched_full','action_gate_first','shuffled_gate_first','uniform_gate_first'][arm%4];policy(panels[arm//4],mode,index//8,8)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage');a.add_argument('index',type=int);a=a.parse_args();run(a.stage,a.index)
