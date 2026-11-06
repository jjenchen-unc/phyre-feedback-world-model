"""Static Slurm chain, with predeclared result-dependent second-round tasks."""
from loss_runtime import *
import argparse

def run(stage,index):
 panels=['cross','within'];kinds=['mse','smooth'];checked()
 if stage=='smoke':
  from smoke import run as f
  f(panels[index]);return
 if stage=='diagnostic':
  from diagnostic import run as f
  f(panels[index]);return
 if stage=='train1':
  from train import run as f
  f(panels[index//2],kinds[index%2],0);return
 if stage=='audit1':
  from audit import run as f
  f(panels[index//2],kinds[index%2],0);return
 if stage=='eval1':
  from evaluate import run as f
  arm=index%4;f(panels[arm//2],kinds[arm%2],0,'raw',index//4,8);return
 if stage=='fixed_new':
  from compare_new import run as f
  f(panels[index//2],kinds[index%2]);return
 if stage=='controls':
  from evaluate import run as f
  arm=index%8;mode=['centered','zero'][arm%2];kind=kinds[(arm//2)%2];panel=panels[arm//4];f(panel,kind,0,mode,index//8,8);return
 # Round2 tasks all exist in queue, but unqualified protocols exit without training/eval.
 arm=index%4 if stage=='eval2' else index;panel=panels[arm//2];choice=read(R/'DECISION.json')['panels'][panel]
 if not choice['replicate']:
  write(R/'skipped'/stage/f'{index}.json',{'skipped':True,'reason':'No qualifying loss in round1','decision':choice});print('SKIP',stage,index);return
 kind='l1' if arm%2==0 else choice['winner']
 if stage=='train2':
  from train import run as f
  f(panel,kind,1)
 elif stage=='audit2':
  from audit import run as f
  f(panel,kind,1)
 elif stage=='eval2':
  from evaluate import run as f
  f(panel,kind,1,'raw',index//4,8)
 else:raise ValueError(stage)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage');a.add_argument('index',type=int);a=a.parse_args();run(a.stage,a.index)
